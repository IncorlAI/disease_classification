import torch.nn as nn
import torch
import torch.nn.functional as F


class SpatialTransformNet(nn.Module):
    def __init__(self, input_size=[94, 24]):
        super(SpatialTransformNet, self).__init__()

        self.input_size = input_size
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
        )

        # Regressor for the 3x2 affine matrix
        w, h = self.input_size
        self.fc_loc = nn.Sequential(nn.Linear(32 * (w//2) * (h//2), 32), nn.Linear(32, 32), nn.Linear(32, 3 * 2))

        # Initialize the weights/bias with identity transformation
        # origin
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        # paper
        # self.fc_loc[1].weight.data.zero_()
        # self.fc_loc[1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        # self.fc_loc[2].weight.data.zero_()
        # self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        batch_size = x.shape[0]
        xs = self.localization(x)
        xs = xs.view(batch_size, -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
