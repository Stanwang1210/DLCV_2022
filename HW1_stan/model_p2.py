import torch
from torch import nn
import torch.nn.functional as F
class Model(nn.Module):

    def __init__(self, model, fcn_model) -> None:
        super().__init__()
        self.feature_extractor = model.features
        self.pool = model.avgpool
        self.fcn = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size = (2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size = (2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=7, kernel_size = (1, 1)),
           
            # nn.Upsample(scale_factor=4, mode='bilinear'),
            # nn.Upsample(scale_factor=2, mode='bilinear'),

        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = self.fcn(x)
        x = F.upsample(x, size=(512, 512), mode='bilinear'),
        return x[0]
    