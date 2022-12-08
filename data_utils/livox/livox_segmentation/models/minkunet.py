# Author: Alexandre Boulch (alexandre.boulch@valeo.com)
# Modified by: Gilles Puy (gilles.puy@valeo.com)
# Original source:
# https://github.com/valeoai/3dssl_surface_recons_dev/blob/dev_lightning/ssl3d_poco/networks/backbone/torchsparse/minkunet.py

import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, stride=stride, transposed=True),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation, stride=1),
            spnn.BatchNorm(outc),
        )

        if inc == outc and stride == 1:
            self.downsample = nn.Sequential()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride),
                spnn.BatchNorm(outc),
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class MinkUNetBase(nn.Module):

    INIT_DIM = 32
    PLANES = (32, 64, 128, 256, 256, 256, 256, 256)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)

    def __init__(self, **kwargs):
        super().__init__()

        in_channels = kwargs["in_channels"]

        self.inplanes = self.INIT_DIM
        l0 = [
            spnn.Conv3d(in_channels, self.inplanes, kernel_size=3, stride=1),
            spnn.BatchNorm(self.inplanes),
            spnn.ReLU(True),
        ]
        self.stem = nn.Sequential(*l0)

        l1 = [
            BasicConvolutionBlock(
                self.inplanes, self.inplanes, ks=2, stride=2, dilation=1
            ),
        ]
        for _ in range(self.LAYERS[0]):
            l1.append(
                ResidualBlock(self.inplanes, self.PLANES[0], ks=3, stride=1, dilation=1)
            )
            self.inplanes = self.PLANES[0]
        self.stage1 = nn.Sequential(*l1)

        l2 = [
            BasicConvolutionBlock(
                self.inplanes, self.inplanes, ks=2, stride=2, dilation=1
            ),
        ]
        for _ in range(self.LAYERS[1]):
            l2.append(
                ResidualBlock(self.inplanes, self.PLANES[1], ks=3, stride=1, dilation=1)
            )
            self.inplanes = self.PLANES[1]
        self.stage2 = nn.Sequential(*l2)

        l3 = [
            BasicConvolutionBlock(
                self.inplanes, self.inplanes, ks=2, stride=2, dilation=1
            ),
        ]
        for _ in range(self.LAYERS[2]):
            l3.append(
                ResidualBlock(self.inplanes, self.PLANES[2], ks=3, stride=1, dilation=1)
            )
            self.inplanes = self.PLANES[2]
        self.stage3 = nn.Sequential(*l3)

        l4 = [
            BasicConvolutionBlock(
                self.inplanes, self.inplanes, ks=2, stride=2, dilation=1
            ),
        ]
        for _ in range(self.LAYERS[3]):
            l4.append(
                ResidualBlock(self.inplanes, self.PLANES[3], ks=3, stride=1, dilation=1)
            )
            self.inplanes = self.PLANES[3]
        self.stage4 = nn.Sequential(*l4)

        u10 = BasicDeconvolutionBlock(self.inplanes, self.PLANES[4], ks=2, stride=2)
        self.inplanes = self.PLANES[4] + self.PLANES[2]
        u11 = []
        for _ in range(self.LAYERS[4]):
            u11.append(
                ResidualBlock(
                    self.inplanes, self.PLANES[4], ks=3, stride=1, dilation=1
                ),
            )
            self.inplanes = self.PLANES[4]
        self.up1 = nn.ModuleList([u10, nn.Sequential(*u11)])

        u20 = BasicDeconvolutionBlock(self.inplanes, self.PLANES[5], ks=2, stride=2)
        self.inplanes = self.PLANES[5] + self.PLANES[1]
        u21 = []
        for _ in range(self.LAYERS[5]):
            u21.append(
                ResidualBlock(
                    self.inplanes, self.PLANES[5], ks=3, stride=1, dilation=1
                ),
            )
            self.inplanes = self.PLANES[5]
        self.up2 = nn.ModuleList([u20, nn.Sequential(*u21)])

        u30 = BasicDeconvolutionBlock(self.inplanes, self.PLANES[6], ks=2, stride=2)
        self.inplanes = self.PLANES[6] + self.PLANES[0]
        u31 = []
        for _ in range(self.LAYERS[6]):
            u31.append(
                ResidualBlock(
                    self.inplanes, self.PLANES[6], ks=3, stride=1, dilation=1
                ),
            )
            self.inplanes = self.PLANES[6]
        self.up3 = nn.ModuleList([u30, nn.Sequential(*u31)])

        u40 = BasicDeconvolutionBlock(self.inplanes, self.PLANES[7], ks=2, stride=2)
        self.inplanes = self.PLANES[7] + self.INIT_DIM
        u41 = []
        for _ in range(self.LAYERS[7]):
            u41.append(
                ResidualBlock(
                    self.inplanes, self.PLANES[7], ks=3, stride=1, dilation=1
                ),
            )
            self.inplanes = self.PLANES[7]
        self.up4 = nn.ModuleList([u40, nn.Sequential(*u41)])

        # default is we create the classifier
        if kwargs["num_classes"] > 0:
            self.classifier = nn.Sequential(
                nn.Linear(self.PLANES[7], kwargs["num_classes"])
            )
        else:
            self.classifier = nn.Identity()

        self.weight_initialization()
        # self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        y1 = self.up1[0](x4)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)

        y3 = self.up3[0](y2)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)

        out = self.classifier(y4.F)

        return out


class MinkUNet34(MinkUNetBase):
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)


class MinkUNet18(MinkUNetBase):
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
