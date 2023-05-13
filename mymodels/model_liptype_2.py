from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import numpy as np
from models.seresnet import se_resnet_34
import options.options_liptype as opt

class LipType(torch.nn.Module):
    def __init__(self, color_mode=0, anno_kind='anno_data_romaji', dropout_p=0.5, dropout3d_p=0.5):
        super(LipType, self).__init__()
        first_channel = 3 if color_mode > 0 else 1
        self.zero_pad3d = nn.ConstantPad3d((2, 2, 2, 2, 1, 1), 0)
        self.conv1 = nn.Conv3d(first_channel, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2)) # conv3d and zeroPad
        self.batc1 = nn.BatchNorm3d(32) # 画像のフレームの長さをとる
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2)) # pool_size, stride

        self.se_net = se_resnet_34()
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.batc2 = nn.BatchNorm2d(512)

        self.gru1  = nn.GRU(512, 256, 1, bidirectional=True)
        self.gru2  = nn.GRU(512, 256, 1, bidirectional=True)
        self.fc_3Dto2D = nn.Linear(opt.vid_padding+4, 1)
        self.fc_2Dto3D = nn.Linear(1, opt.vid_padding+4)

        self.fc = None
        if anno_kind == 'anno_data_romaji':
            self.fc    = nn.Linear(512, 27+1)
        elif anno_kind == 'anno_data_mydic':
            self.fc = nn.Linear(512, 25+1)
        # self.gru1_2 = nn.GRU(40, 256, 1, bidirectional=True)
        # self.gru2_2 = nn.GRU(512, 256, 1, bidirectional=True)

        self.dropout_p  = dropout_p
        self.dropout3d_p = dropout3d_p

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)
        self.dropout3d = nn.Dropout3d(self.dropout3d_p)
        self._init()

    def _init(self):

        init.kaiming_normal_(self.conv1.weight, nonlinearity='relu') # He初期化 重みの初期化
        init.constant_(self.conv1.bias, 0) # 入力テンソルに値を入れる constant_(テンソル, 埋める値) バイアスの初期化
        init.kaiming_normal_(self.fc_2Dto3D.weight, nonlinearity='relu')
        init.constant_(self.fc_3Dto2D.bias, 0)
        init.kaiming_normal_(self.fc_3Dto2D.weight, nonlinearity='relu')
        init.constant_(self.fc_2Dto3D.bias, 0)

        init.kaiming_normal_(self.fc.weight, nonlinearity='sigmoid')
        init.constant_(self.fc.bias, 0)

        stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
        for m in (self.gru1, self.gru2):#, self.gru3):
            #stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
            for i in range(0, 256 * 3, 256):
                init.uniform_(m.weight_ih_l0[i: i + 256],
                            -math.sqrt(3) * stdv, math.sqrt(3) * stdv) # 入力されたテンソルを一様分布から引き出された値で埋める
                #init.uniform_(m.weight_ih_l0[i: i + 256],
                            #-math.sqrt(3) * stdv, math.sqrt(3) * stdv) # 入力されたテンソルを一様分布から引き出された値で埋める

                # 入力テンソルを(半)直交行列で埋める
                # また、入力するテンソルは少なくとも2次元は必要であり、2次元を超える場合、後続の次元は平坦化される。
                # 重みの初期化

                init.orthogonal_(m.weight_hh_l0[i: i + 256])
                # バイアスの初期化
                init.constant_(m.bias_ih_l0[i: i + 256], 0)

                # 上記と同じ
                init.uniform_(m.weight_ih_l0_reverse[i: i + 256],
                            -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0_reverse[i: i + 256])
                init.constant_(m.bias_ih_l0_reverse[i: i + 256], 0)

                #init.orthogonal_(m.weight_hh_l0[i: i + 256])
                # バイアスの初期化
                #init.constant_(m.bias_ih_l0[i: i + 256], 0)

                # 上記と同じ
                #init.uniform_(m.weight_ih_l0_reverse[i: i + 256],
                            #-math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                #init.orthogonal_(m.weight_hh_l0_reverse[i: i + 256])
                #init.constant_(m.bias_ih_l0_reverse[i: i + 256], 0)
        # stdv = math.sqrt(2 / (40 + 256))
        # for m in (self.gru1_2, self.gru2_2):#, self.gru3):
        #     #stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
        #     for i in range(0, 256 * 3, 256):
        #         init.uniform_(m.weight_ih_l0[i: i + 256],
        #                     -math.sqrt(3) * stdv, math.sqrt(3) * stdv) # 入力されたテンソルを一様分布から引き出された値で埋める
        #         #init.uniform_(m.weight_ih_l0[i: i + 256],
        #                     #-math.sqrt(3) * stdv, math.sqrt(3) * stdv) # 入力されたテンソルを一様分布から引き出された値で埋める

        #         # 入力テンソルを(半)直交行列で埋める
        #         # また、入力するテンソルは少なくとも2次元は必要であり、2次元を超える場合、後続の次元は平坦化される。
        #         # 重みの初期化

        #         init.orthogonal_(m.weight_hh_l0[i: i + 256])
        #         # バイアスの初期化
        #         init.constant_(m.bias_ih_l0[i: i + 256], 0)

        #         # 上記と同じ
        #         init.uniform_(m.weight_ih_l0_reverse[i: i + 256],
        #                     -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
        #         init.orthogonal_(m.weight_hh_l0_reverse[i: i + 256])
        #         init.constant_(m.bias_ih_l0_reverse[i: i + 256], 0)


    def forward(self, x):
        x = self.zero_pad3d(x)
        x = self.conv1(x)
        x = self.batc1(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool1(x)

        x = self.zero_pad3d(x)
        # batc_len = x.size(0)
        # frame_len = x.size(2)
        
        # 3D to 2D
        x = x.permute(0, 1, 3, 4, 2)
        x = self.fc_3Dto2D(x).contiguous()
        x = self.relu(x)
        x = x.squeeze(-1)
        
        x = x.contiguous()
        x = self.se_net(x)
        x = self.batc2(x)
        x = self.relu(x)
        # 2D to 3D
        x = x.unsqueeze(-1)
        x = self.fc_2Dto3D(x)
        x = self.relu(x)
        x = x.permute(0, 1, 4, 2, 3)
        x = self.dropout3d(x)
        # (B, C, T, H, W)->(T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4).contiguous() # 軸の順番を変更
        # また、contiguous()はviewにするとき、メモリ上に要素順に並ぶため、エラーを回避できる
        # (T, B, C, H, W)->(T, B, C*H*W)
        x = x.view(x.size(0), x.size(1), -1).contiguous()
        
        # RNNの重みがメモリ上で非連続にならないように
        # また、パラメータデータポインターをリセットして、より高速なコードパスを使用できるようになっている。
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
        x, h = self.gru1(x)
        x, h = self.gru2(x)
        x = x.view(x.size(1), x.size(0), -1).contiguous()

        # pad = torch.zeros(dim.size(0), 2, dim.size(2), dim.size(3)).cuda()
        # dim = torch.cat([pad, dim, pad], dim=1)
        # dim = dim.view(dim.size(1), dim.size(0), -1).contiguous()
        # self.gru1_2.flatten_parameters()
        # self.gru2_2.flatten_parameters()
        # dim, hd = self.gru1_2(dim)
        # dim, hd = self.gru2_2(dim)
        # dim = dim.permute(1, 0, 2).contiguous()
        # #dim = self.fc_2(dim) # (T, B, H)
        # #dim = dim.permute(1, 0, 2).contiguous() # (B, T, H)
        # x = torch.add(x, dim)
        x = self.fc(x)
        x = x.permute(1, 0, 2).contiguous()
        return x

if __name__ == '__main__':
    model = LipType()
    data = torch.randn(2, 1, 925, 64, 128)
    #dim = torch.randn(1, 925, 20, 2)
    out = model(data)
    print(out.size())