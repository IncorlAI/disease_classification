from torchvision.models.resnet import resnet50
import torch.nn as nn

class Architecture_tmp(nn.Module):
    def __init__(self, model_type, num_classes):
        super(Architecture_tmp, self).__init__()
        if model_type == 'resnet':
            self.model = resnet50(pretrained=True)
            self.model.fc = nn.Linear(2048, out_features=num_classes)
        
    def forward(self, x):
        output = self.model(x)
        return output