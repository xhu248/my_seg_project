import torch
from torch import nn

class ExtractNN(nn.Module):
    def _init_(self, num_classes=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=num_classes, out_channels=1, kernel_size=1)
        self.fc = nn.linear(512*512, 1)

    def forward(self, x):
        x = self.conv1(x)
        output = self.fc(x)

        return output