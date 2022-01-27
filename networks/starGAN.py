import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_features),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, img_shape=(3, 256, 256), res_blocks=9, c_dim=3):
        super(GeneratorResNet, self).__init__()
        channels, img_size, _ = img_shape

        # Initial convolution block
        curr_dim = 14*4
        model = [
            nn.Conv2d(channels + c_dim, curr_dim, 7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(curr_dim),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        for _ in range(2):
            model += [
                nn.Conv2d(curr_dim, curr_dim * 2, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(curr_dim * 2),
                nn.ReLU(inplace=True),
            ]
            curr_dim *= 2

        # Residual blocks
        for _ in range(res_blocks):
            model += [ResidualBlock(curr_dim)]

        self.encoder = nn.Sequential(*model)
        
        model = []
        # Upsampling
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(curr_dim // 2),
                nn.ReLU(inplace=True),
            ]
            curr_dim = curr_dim // 2

        # Output layer
        model += [nn.Conv2d(curr_dim, channels, 7, stride=1, padding=3), nn.Tanh()]

        self.decoder = nn.Sequential(*model)

    def forward(self, x, rand_dise, rand_risk, device='cuda:0'):
        b, c, w, h = x.shape
        encoded = self.encoder(x)
        corr = torch.full(size=(b, 14 * 4 * 4), fill_value=0, dtype=torch.float, device=device)
        for i in range(b):
            dise_range = rand_dise[i] * 4 * 4
            risk_range = rand_risk[i] * 4
            start = dise_range + risk_range
            end = start + 4
            corr[i, start:end] = 1.0
        latent = encoded * corr.reshape(b, 14 * 4 * 4, 1, 1)
        # output = (self.decoder(encoded) + 1) / 2 + x
        output = self.decoder(latent) + x
        output = torch.clip(output, min=0, max=1)
        return output


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, img_shape=(3, 128, 128), c_dim=5, n_strided=3):
        super(Discriminator, self).__init__()
        channels, img_size, _ = img_shape

        def discriminator_block(in_filters, out_filters):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1), nn.LeakyReLU(0.01)]
            return layers

        layers = discriminator_block(channels, 64)
        curr_dim = 64
        for _ in range(n_strided - 1):
            layers.extend(discriminator_block(curr_dim, curr_dim * 2))
            curr_dim *= 2

        self.model = nn.Sequential(*layers)

        # Output 1: PatchGAN
        self.out1 = nn.Conv2d(curr_dim, 1, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, img):
        feature_repr = self.model(img)
        out_adv = self.out1(feature_repr)
        out_adv = self.sigmoid(out_adv)
        return out_adv