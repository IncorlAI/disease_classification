from torch import nn
import torchvision
from efficientnet_pytorch import EfficientNet


class Efficient(nn.Module):
    def __init__(self, crop=6, dise=14, risk=4):
        super(Efficient, self).__init__()
        self.EffNet = EfficientNet.from_pretrained('efficientnet-b7')

        # self.EffNet = nn.Sequential(*list(EffNet.children())[:-1])
        self.crop_out = nn.Linear(2048, crop)
        self.dise_out = nn.Linear(2048, dise)
        self.risk_out = nn.Linear(2048, risk)

    def forward(self, input):
        batch_size = input.shape[0]
        encoded = self.EffNet(input)  # (N, feature_map_channels, feature_map_w, feature_map_h)
        encoded = encoded.view(batch_size, -1)
        crop = self.crop_out(encoded)
        dise = self.dise_out(encoded)
        risk = self.risk_out(encoded)

        return (crop, dise, risk)
