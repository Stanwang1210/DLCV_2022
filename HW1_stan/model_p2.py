import torch
from torch import nn
import torch.nn.functional as F
class Model(nn.Module):

    def __init__(self, model) -> None:
        super().__init__()
        self.feature_extractor = model.features
        self.fcn = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size = (1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size = (1, 1)),
            nn.ReLU(),
            # nn.Conv2d(in_channels=32, out_channels=7, kernel_size = (1, 1)),
           
            # nn.Upsample(size=(128, 128), mode='bilinear'),
            nn.ConvTranspose2d(in_channels=256, out_channels=7, kernel_size=32, stride=32, dilation=1)  # [in_channel, out_channel, kernel_size, stride]
            # nn.Upsample(scale_factor=2, mode='bilinear'),

        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fcn(x)
        return x
    