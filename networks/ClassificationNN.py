import torch
from torch import nn
import torch.nn.functional as F


class ClassificationNN(nn.Module):

    def __init__(self, num_classes=2):
        super(ClassificationNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.global_max = nn.MaxPool2d(kernel_size=64)
        self.global_avg = nn.AvgPool2d(kernel_size=64)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = self.global_max(x)
        x2 = self.global_avg(x)
        x = torch.cat([x1, x2], 1)
        output = self.fc(x.squeeze()).reshape(4, 2)

        return output