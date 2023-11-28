import torch.nn as nn
import torch
import torch.nn.functional as F

torch.backends.cudnn.deterministic = True
torch.set_default_dtype(torch.float64)
torch.manual_seed(42)


class LeNet5(nn.Module):
    def __init__(
        self, num_classes: int, input_channels: int, activate_bn: bool = False
    ):
        super(LeNet5, self).__init__()
        if activate_bn:
            self.layer1 = nn.Sequential(
                nn.Conv2d(input_channels, 6, kernel_size=5, stride=1, padding=0),
                nn.BatchNorm2d(6),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(input_channels, 6, kernel_size=5, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

        if activate_bn:
            self.layer2 = nn.Sequential(
                nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        else:
            self.layer2 = nn.Sequential(
                nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


class FCN3(nn.Module):
    def __init__(self, num_classes, input_features):
        super(FCN3, self).__init__()
        self.fc1 = nn.Linear(input_features, 1000)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1000, 100)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(100, num_classes)

    def forward(self, x):
        out = x.reshape(x.size(0), -1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=True,
                )
            )

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class ResidualBlock(nn.Module):
    def __init__(self, n_features):
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(n_features, n_features, 3, 1, 1),
            # nn.BatchNorm2d(n_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_features, n_features, 3, 1, 1),
            # nn.BatchNorm2d(n_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_block(x) + x


class Resnet(nn.Module):
    def __init__(self, in_ch=3, out_ch=128, n_blocks=4):
        super(Resnet, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.n_blocks = n_blocks

        # input size: in_ch x 32 x 32

        ## Features ##
        n_features = 64

        # 1st conv:
        features = [
            nn.Conv2d(in_ch, n_features, 4, 2, 1),
            # nn.BatchNorm2d(n_features),
            nn.ReLU(inplace=True),
        ]

        for i in range(self.n_blocks):
            features += [ResidualBlock(n_features)]

        features += [nn.Conv2d(n_features, out_ch, 4, 2, 1), nn.ReLU(inplace=True)]

        self.features = nn.Sequential(*features)
        # state size: out_ch x 8 x 8

        ## Classifier ##
        classifier = [
            nn.Linear(self.out_ch * 8 * 8, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10),
        ]

        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.out_ch * 8 * 8)
        x = self.classifier(x)

        return x


class VGG_mini(nn.Module):

    def __init__(self):
        super(VGG_mini, self).__init__()

        # Maxpool 2x2

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv layers with batch norm
        self.conv1 = nn.Conv2d(3, 64, 3, padding = 1)

        self.conv2 = nn.Conv2d(64, 64, 3, padding = 1)

        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)

        self.conv4 = nn.Conv2d(128, 128, 3, padding = 1)

        self.conv5 = nn.Conv2d(128, 256, 3, padding = 1)

        self.conv6 = nn.Conv2d(256, 256, 3, padding = 1)

        self.conv7 = nn.Conv2d(256, 512, 3, padding = 1)

        self.conv8 = nn.Conv2d(512, 512, 3, padding = 1)

        # fully connected layer with batch norm

        self.fc1 = nn.Linear(512 * 4 * 4, 128)

        self.fc2 = nn.Linear(128, 64)

        self.fc3 = nn.Linear(64, 10)


    def forward(self, x):

        # Difference from ref. implementation:
        # - Removed the batch norm layers
        out = F.elu(self.conv1(x))
        out = F.elu(self.conv2(out))
        out = self.pool(out)

        out = F.elu(self.conv3(out))
        out = F.elu(self.conv4(out))
        out = self.pool(out)

        out = F.elu(self.conv5(out))
        out = F.elu(self.conv6(out))
        out = self.pool(out)

        out = F.elu(self.conv7(out))
        out = F.elu(self.conv8(out))

        out = out.view(-1, 512 * 4 * 4)

        out = F.elu(self.fc1(out))
        out = F.elu(self.fc2(out))
        out = self.fc3(out)

        return out