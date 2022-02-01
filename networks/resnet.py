from cv2 import norm
from torch import nn
import torchvision


class ResNet50(nn.Module):
    # def __init__(self, crop=6, dise=14, risk=4):
    def __init__(self, num_output):
        super(ResNet50, self).__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Linear(2048, out_features=num_output) 
        # self.resnet50 = nn.Sequential(*list(resnet50.children())[:-1])
        # self.risk_out = nn.Linear(2048, num_output)
        # self.crop_out = nn.Linear(2048, crop)
        # self.dise_out = nn.Linear(2048, dise)
        # self.risk_out = nn.Linear(2048, risk)
        a=1
    def forward(self, x):
        output = self.resnet50(x)
        # batch_size = input.shape[0]
        # encoded = self.resnet50(input)  # (N, feature_map_channels, feature_map_w, feature_map_h)
        # encoded = encoded.view(batch_size, -1)

        # output_risk = self.risk_out(encoded)
        return output