import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class SEBasicBlock(nn.Module):
    expansion = 1
    def __init__(
        self,
        in_channels, 
        out_channels, 
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        stride=1,
        r=16
    ):
        super(SEBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = norm_layer(in_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=3, padding=1, stride=1)
        self.bn2 = norm_layer(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels*self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*self.expansion, 1, stride=stride),
                nn.BatchNorm2d(out_channels*self.expansion)
            )
        
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels, r)
        self._init_weights()
    
    def _init_weights(self):
        init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        init.constant_(self.conv1.bias, 0)
        init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        init.constant_(self.conv2.bias, 0)
        if self.downsample is not None:
            for m in self.downsample:
                if m._get_name() == 'Conv2d':
                    init.kaiming_normal_(m.weight)
                    init.constant_(m.bias, 0)
    
    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # C x H x W -> C x 1 x 1
        self.fc1 = nn.Linear(c, c//r, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(c//r, c, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x) # expand_as(x) yをxと同じサイズに拡張


class SENet(nn.Module):
    def __init__(self, input_size, block, layers, pooling=None, include_top=False, num_classes=1000):
        super(SENet, self).__init__()
        self.in_channels=64
        self.conv1 = nn.Conv2d(32, 64, kernel_size=7, stride=2,
            padding=[((2*(input_size[0]-1)-input_size[0]+7)//2), ((2*(input_size[1]-1)-input_size[1]+7)//2)])
        self.bn1 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(3, stride=2)
        self.padding1 = nn.ZeroPad2d((9, 9, 5, 5))

        self.layer1 = self._make_layer(block, 64, layers[0])
        #input_size  = [input_size[0]//2, input_size[1]//2]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        #input_size  = [input_size[0]//2, input_size[1]//2]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        #input_size  = [input_size[0]//2, input_size[1]//2]
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #input_size  = [input_size[0]//2, input_size[1]//2]
        self.avgpool = nn.AvgPool2d(1, stride=1)
        self.maxpool = nn.MaxPool2d(1, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.sigmoid = nn.Sigmoid()
        self.include_top = include_top
        self.pooling = pooling
        self._init_weights()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride=stride))
        self.in_channels = out_channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        init.constant_(self.conv1.bias, 0)

        init.kaiming_normal_(self.fc.weight, nonlinearity='sigmoid')
        init.constant_(self.fc.bias, 0)


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.padding1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn1(x)
        x = self.relu(x)
        if self.include_top:
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            x = self.sigmoid(x)
        else:
            if self.pooling == 'avg':
                x = F.adaptive_avg_pool2d(x, 1)
                # x = x.view(x.size(0), -1)
            elif self.pooling == 'max':
                x = F.adaptive_max_pool2d(x, 1)
                # x = x.view(x.size(0), -1)
        return x

def se_resnet_34(num_classes=1000, pretrained=False, **kwargs):
    model = SENet([17, 33], block=SEBasicBlock, layers=[3,4,6,3], 
                  include_top=False, pooling='avg', num_classes=num_classes)
    return model


if __name__ == '__main__':
    
    # データ作成
    input= torch.randn(2, 3, 240, 64, 128)
    pad1 = nn.ConstantPad3d((2,2,2,2,1,1), 0)
    conv1 = nn.Conv3d(3, 32, (3,5,5), (1,2,2), (1,2,2))
    batc1 = nn.BatchNorm3d(32)
    relu = nn.ReLU(inplace=True)
    pool1 = nn.MaxPool3d((1,2,2), (1,2,2))
    out1 = pad1(input)
    out1 = conv1(out1)
    out1 = batc1(out1)
    out1 = relu(out1)
    out1 = pool1(out1)
    pad2 = nn.ConstantPad3d((2,2,2,2,1,1), 0)
    out1 = pad2(out1)
    print(out1.size())
    outx = out1.permute(0, 2, 1, 3, 4)
    outx = torch.reshape(outx, (outx.size(0)*outx.size(1), outx.size(2), outx.size(3), outx.size(4)))
    se_net = se_resnet_34()
    out2 = se_net(outx)
    print(out2.size())
    """
    print(outx.size())
    conv2 = nn.Conv2d(32, 64, 7, 2, padding=[((2*(outx.size(2)-1)-outx.size(2)+7)//2), ((2*(outx.size(3)-1)-outx.size(3)+7)//2)])
    out2 = conv2(outx)
    print(out2.size())
    maxpool2 = nn.MaxPool2d(3, stride=2)#, padding=[((2*((out2.size(2)//2)-1)-17+3)//2)-1, ((2*((out2.size(3)//2)-1)-33+3)//2)-1])
    out2 = maxpool2(out2)
    print(out2.size())
    exit()
    input_size = [out2.size(2), out2.size(3)]
    print(out2.size())
    se_basic = SEBasicBlock(input_size, out2.size(1), 64, stride=2)
    result = se_basic.forward(out2)
    print(result.size())"""

"""
    def __init__(
        self,
        input_size,
        in_channels, 
        out_channels, 
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        stride=1,
        r=16
    ):

"""