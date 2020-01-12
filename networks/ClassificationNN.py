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
        # self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = self.global_max(x)
        x2 = self.global_avg(x)
        x = torch.cat([x1, x2], 1)
        # x = torch.flatten(x, 1)
        # output = self.fc(x.squeeze())

        return x


class ClassificationUnet(nn.Module):
    def __init__(self, num_classes=2, in_channels=1, initial_filter_size=64, kernel_size=3, num_downs=3, norm_layer=nn.InstanceNorm3d):
        super(ClassificationUnet, self).__init__()

        self.num_classes = num_classes

        block = DownsamplingBlock(in_channels=initial_filter_size * 2 ** (num_downs-1), out_channels=initial_filter_size * 2 ** num_downs,
                                             kernel_size=kernel_size, norm_layer=norm_layer)

        for i in range(1, num_downs):
            block = DownsamplingBlock(in_channels=initial_filter_size * 2 ** (num_downs-(i+1)),
                                      out_channels=initial_filter_size * 2 ** (num_downs-i),
                                      kernel_size=kernel_size, submodule=block, norm_layer=norm_layer)

        block = DownsamplingBlock(in_channels=in_channels, out_channels=initial_filter_size,
                                             kernel_size=kernel_size, submodule=block, norm_layer=norm_layer)

        self.model = block
        self.fc = nn.Linear(initial_filter_size * 2 ** (num_downs+1), num_classes)

    @staticmethod
    def pooling(layer, in_channels=None, kernel_size=None):
        global_max = nn.MaxPool3d(kernel_size=kernel_size)
        global_avg = nn.AvgPool3d(kernel_size=kernel_size)
        layer = torch.cat([global_avg(layer), global_max(layer)], 1)
        layer = torch.flatten(layer, 1)
        return layer

    def forward(self, x):
        x = self.model(x)
        x = self.pooling(x, in_channels=x.size()[1], kernel_size=x.size()[2])

        y = self.fc(x)

        return x, y


# half part of unet, the extraction path without skipping connection
class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, kernel_size=3, submodule=None, norm_layer=nn.InstanceNorm3d):
        super(DownsamplingBlock, self).__init__()

        pool = nn.MaxPool3d(2, stride=2)
        conv1 = self.contract(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              norm_layer=norm_layer)
        conv2 = self.contract(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                              norm_layer=norm_layer)
        if submodule is None:
            down = [conv1, conv2]
            model = down
        else:
            down = [conv1, conv2, pool]
            model = down + [submodule]

        self.model = nn.Sequential(*model)

    @staticmethod
    def contract(in_channels, out_channels, kernel_size=3, norm_layer=nn.InstanceNorm3d):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=1),
            norm_layer(out_channels),
            nn.LeakyReLU(inplace=True))
        return layer

    def forward(self, x):
        return self.model(x)
