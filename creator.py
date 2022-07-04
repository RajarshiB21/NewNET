import torch
import torch.nn as nn


class Creator(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Creator, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.inception1 = InceptionCombiner(192, 64, 96, 128, 16, 32, 32, 1)#number of channels of output shape = 256
        self.conv3 = ConvCreator(256, 256, 3, 1, 1)
        self.conv4 = ConvCreator(256, 256, 3, 1, 1)
        self.inception2 = InceptionCombiner(256, 128, 128, 192, 32, 96, 64, 3)#number of channels of output shape = 1440
        self.conv5 = ConvCreator(1440, 1440, 3, 1, 1)
        self.conv6 = ConvCreator(1440, 1440, 3, 1, 1)
        self.conv7 = ConvCreator(1440, 1440, 3, 1, 1)
        self.inception3 = InceptionCombiner(1440, 192, 96, 208, 16, 48, 64, 5)#number of channels of output shape = 2560
        self.conv8 = ConvCreator(2560, 2560, 3, 1, 1)
        self.conv9 = ConvCreator(2560, 2560, 3, 1, 1)
        self.fc1 = nn.Linear(2560*15*15, 1000)
        self.fc2 = nn.Linear(1000, 400)
        self.fc3 = nn.Linear(400, num_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.inception1(x)#226

        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.inception2(x)#115

        x = self.conv5(x)
        x = self.pool(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.inception3(x)

        x = self.conv8(x)
        x = self.pool(x)
        x = self.conv9(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)

        return x

class ConvCreator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvCreator, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))


class Inception(nn.Module):
    def __init__(self, in_channels, out1x1, red3x3, out3x3, red5x5, out5x5, out1x1pool):
        super(Inception, self).__init__()
        self.branch1 = ConvCreator(in_channels, out1x1, kernel_size=1, stride=1, padding=1)
        self.branch2 = nn.Sequential(
            ConvCreator(in_channels, red3x3, kernel_size=1, stride=1, padding=1),
            ConvCreator(red3x3, out3x3, kernel_size=3, stride=1, padding=1)
        )
        self.branch3 = nn.Sequential(
            ConvCreator(in_channels, red5x5, kernel_size=1, stride=1, padding=1),
            ConvCreator(red5x5, out5x5, kernel_size=5, stride=1, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvCreator(in_channels, out1x1pool, kernel_size=1, stride=1, padding=1)
        )

    def forward(self, x):#1, 3, 224, 224
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)


class InceptionCombiner(nn.Module):
    def __init__(self, in_channels, out1x1, red3x3, out3x3, red5x5, out5x5, out1x1pool,x1):
        super(InceptionCombiner, self).__init__()
        self.x1 = x1
        self.bbranch1 = Inception(in_channels, out1x1, red3x3, out3x3, red5x5, out5x5, out1x1pool)

        self.bbranch3a = Inception(in_channels, out1x1, red3x3, out3x3, red5x5, out5x5, out1x1pool)
        self.bbranch3b = Inception(in_channels, out1x1, red3x3, out3x3, red5x5, out5x5, out1x1pool)
        self.bbranch3c = Inception(in_channels, out1x1, red3x3, out3x3, red5x5, out5x5, out1x1pool)

        self.bbranch5a = Inception(in_channels, out1x1, red3x3, out3x3, red5x5, out5x5, out1x1pool)
        self.bbranch5b = Inception(in_channels, out1x1, red3x3, out3x3, red5x5, out5x5, out1x1pool)
        self.bbranch5c = Inception(in_channels, out1x1, red3x3, out3x3, red5x5, out5x5, out1x1pool)
        self.bbranch5d = Inception(in_channels, out1x1, red3x3, out3x3, red5x5, out5x5, out1x1pool)
        self.bbranch5e = Inception(in_channels, out1x1, red3x3, out3x3, red5x5, out5x5, out1x1pool)


    def forward(self, x):
        if self.x1 == 1:
            return self.bbranch1(x)

        if self.x1 == 3:
            return torch.cat([self.bbranch3a(x), self.bbranch3b(x), self.bbranch3c(x)], 1)

        if self.x1 == 5:
            return torch.cat([self.bbranch5a(x), self.bbranch5b(x), self.bbranch5c(x), self.bbranch5d(x), self.bbranch5e(x)], 1)


def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn((1, 3, 224, 224)).to(device)
    model = Creator(in_channels=3,num_classes=10).to(device)
    print(model(x).shape)


if __name__ == "__main__":
    test()






