import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import numpy as np


class CNN3D_x2(torch.nn.Module):
    def __init__(self, color_mode, dropout_p=0.30, is_grad=False):  # color_mode -> 0: gray or 1: RGB
        super(CNN3D_x2, self).__init__()
        self.is_grad = is_grad
        first_feature = 3 if color_mode else 1
        self.conv1 = nn.Conv3d(first_feature, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.dropout_p  = dropout_p

        self.relu = nn.ReLU(inplace=True)
        self.dropout3d = nn.Dropout3d(self.dropout_p)
        self._init()

    def _init(self):

        init.kaiming_normal_(self.conv1.weight, nonlinearity='relu') # He初期化 重みの初期化
        init.constant_(self.conv1.bias, 0) # 入力テンソルに値を入れる constant_(テンソル, 埋める値) バイアスの初期化

        #以下上記と同じ
        init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        init.constant_(self.conv2.bias, 0)

        init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        init.constant_(self.conv3.bias, 0)


    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool3(x)
        if self.is_grad:
            x = x.clone().detach().requires_grad_(True)
        # (B, C, T, H, W)->(T, B, C, H, W)
        out = x
        if self.is_grad:
            return out, x
        return out

if __name__ == '__main__':
    inputs = torch.randn(1, 1, 179, 64, 128)
    model = CNN3D_x2(color_mode=0)
    print(model)