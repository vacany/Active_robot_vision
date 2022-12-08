import torch.nn as nn
import torchsparse.nn as spnn


class MiniPointNet(nn.Module):

    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.bn = spnn.BatchNorm(in_channels)
        self.layer1 = nn.Linear(in_channels, 128)
        self.relu = nn.ReLU(inplace=True)
        self.layer2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.bn(x).F
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
