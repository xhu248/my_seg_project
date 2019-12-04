import torch
from torch import nn


class DownsamplingResBlock(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, kernel_size=3, submodule=None,
                 outermost=False, norm_layer=nn.InstanceNorm3d):
        super(DownsamplingResBlock, self).__init__()

        self.submodule = submodule
        self.outermost = outermost
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool = nn.MaxPool3d(2, stride=2)

        conv1 = self.contract(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              norm_layer=norm_layer)
        conv2 = self.contract(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                              norm_layer=norm_layer)

        model = [conv1, conv2]

        self.shortcut = self.bypass(in_channels, out_channels)

        self.model = nn.Sequential(*model)

    @staticmethod
    def contract(in_channels, out_channels, kernel_size=3, norm_layer=nn.InstanceNorm3d):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=1),
            norm_layer(out_channels),
            nn.LeakyReLU(inplace=True))
        return layer

    @staticmethod
    def bypass(in_channels, out_channels):
        conv_bypass = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        return conv_bypass

    def forward(self, x):
        if self.submodule is None:
            y = self.model(x)
            y = y + self.shortcut(x)
        else:
            y = self.model(x)
            y = y + self.shortcut(x)
            y = self.pool(y)
            y = self.submodule(y)

        return y


class ClassificationVnet(nn.Module):
    def __init__(self, num_classes=2, in_channels=1, initial_filter_size=64, kernel_size=3,
                 num_downs=3, norm_layer=nn.InstanceNorm3d):
        super(ClassificationVnet, self).__init__()

        self.num_classes = num_classes

        block = DownsamplingResBlock(in_channels=initial_filter_size * 2 ** (num_downs - 1),
                                     out_channels=initial_filter_size * 2 ** num_downs,
                                     kernel_size=kernel_size, norm_layer=norm_layer)

        for i in range(1, num_downs):
            block = DownsamplingResBlock(in_channels=initial_filter_size * 2 ** (num_downs - (i + 1)),
                                         out_channels=initial_filter_size * 2 ** (num_downs - i),
                                         kernel_size=kernel_size, submodule=block, norm_layer=norm_layer)

        block = DownsamplingResBlock(in_channels=in_channels, out_channels=initial_filter_size,
                                     kernel_size=kernel_size, submodule=block, norm_layer=norm_layer)

        self.model = block
        self.fc = nn.Linear(initial_filter_size * 2 ** (num_downs + 1), num_classes)

    @staticmethod
    def pooling(layer, kernel_size=None):
        global_max = nn.MaxPool3d(kernel_size=kernel_size)
        global_avg = nn.AvgPool3d(kernel_size=kernel_size)
        layer = torch.cat([global_avg(layer), global_max(layer)], 1)
        layer = torch.flatten(layer, 1)
        return layer

    def forward(self, x):
        x = self.model(x)
        x = self.pooling(x, kernel_size=x.size()[2])

        x = self.fc(x)
        return x
