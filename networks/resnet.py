from torch import nn
import torchvision


class ResNet50(nn.Module):
    def __init__(self, crop=6, dise=14, risk=4):
        super(ResNet50, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        
        self.resnet50 = nn.Sequential(*list(resnet50.children())[:-1])
        self.crop_out = nn.Linear(2048, crop)
        self.dise_out = nn.Linear(2048, dise)
        self.risk_out = nn.Linear(2048, risk)

    def forward(self, input):
        batch_size = input.shape[0]
        encoded = self.resnet50(input)  # (N, feature_map_channels, feature_map_w, feature_map_h)
        encoded = encoded.view(batch_size, -1)
        crop = self.crop_out(encoded)
        dise = self.dise_out(encoded)
        risk = self.risk_out(encoded)

        return (crop, dise, risk)
    
    
if __name__ == "__main__":
    resnet = TruncatedResNet18()
    print("test")