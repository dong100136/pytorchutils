import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResidualBottlenect(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBottlenect, self).__init__()
        temp_channel = int(inchannel//4)
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, temp_channel, kernel_size=1,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(temp_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(temp_channel, temp_channel, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(temp_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(temp_channel, outchannel, kernel_size=1,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, init_layer, layers_config, num_classes=5,include_top = True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.inchannel = init_layer['outchannel']
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, init_layer['outchannel'], kernel_size=init_layer['kernel_size'],
                      stride=init_layer['stride'], padding=1, bias=False),
            nn.BatchNorm2d(init_layer['outchannel']),
            nn.ReLU(),
        )
        self.layers = nn.ModuleDict()
        for i, l in enumerate(layers_config):
            layer = self.make_layer(ResidualBlock, l[1], l[0], stride=l[2])
            self.layers['layer_%d' % i] = layer
        self.fc = nn.Linear(layers_config[-1][1], num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        for i in range(len(self.layers)):
            # print(i, self.layers['layer_%d' % i])
            out = self.layers['layer_%d' % i](out)
        out = F.adaptive_avg_pool2d(out, (1))
        out = out.view(out.size(0), -1)

        if self.include_top:
            out = self.fc(out)
        return out


def ResNet18(num_classes=5,**kwargs):
    init_layer = {'kernel_size': 7, 'outchannel': 64, 'stride': 2}
    layers = [(2, 64, 2), (2, 128, 2), (2, 256, 2), (2, 512, 2)]
    return ResNet(ResidualBlock, init_layer, layers, num_classes=num_classes,**kwargs)


def ResNet34(num_classes=5,include_top=True):
    init_layer = {'kernel_size': 7, 'outchannel': 64, 'stride': 2}
    layers = [(3, 64, 2), (4, 128, 2), (6, 256, 2), (3, 512, 2)]
    return ResNet(ResidualBlock, init_layer, layers, num_classes=num_classes, include_top=True)


def ResNet50(num_classes=5,**kwargs):
    init_layer = {'kernel_size': 7, 'outchannel': 64, 'stride': 2}
    layers = [(3, 256, 2), (4, 512, 2), (6, 1024, 2), (3, 2048, 2)]
    return ResNet(ResidualBottlenect, init_layer, layers, num_classes=num_classes,**kwargs)


def ResNet101(num_classes=5):
    init_layer = {'kernel_size': 7, 'outchannel': 64, 'stride': 2}
    layers = [(3, 256, 2), (4, 512, 2), (23, 1024, 2), (3, 2048, 2)]
    return ResNet(ResidualBottlenect, init_layer, layers, num_classes=num_classes)


def ResNet152(num_classes=5):
    init_layer = {'kernel_size': 7, 'outchannel': 64, 'stride': 2}
    layers = [(3, 256, 2), (8, 512, 2), (36, 1024, 2), (3, 2048, 2)]
    return ResNet(ResidualBottlenect, init_layer, layers, num_classes=num_classes)


def Cifar10_ResNet20(num_classes=10):
    init_layer = {'kernel_size': 3, 'outchannel': 16, 'stride': 1}
    layers = [(3, 16, 1), (3, 32, 2), (3, 64, 2)]
    return ResNet(ResidualBlock, init_layer, layers, num_classes=num_classes)


def Cifar10_ResNet34(num_classes=10):
    init_layer = {'kernel_size': 3, 'outchannel': 16, 'stride': 1}
    layers = [(11, 16, 1), (10, 32, 2), (10, 64, 2)]
    return ResNet(ResidualBlock, init_layer, layers, num_classes=num_classes)


def Cifar10_ResNet44(num_classes=10):
    init_layer = {'kernel_size': 3, 'outchannel': 32, 'stride': 1}
    layers = [(17, 32, 1), (16, 64, 2), (16, 128, 2)]
    return ResNet(ResidualBlock, init_layer, layers, num_classes=num_classes)
