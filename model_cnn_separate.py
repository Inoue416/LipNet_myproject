import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import numpy as np
from models.cnn3d_separate import CNN3D_x2


class LipNetCNNSeparate(torch.nn.Module):
    def __init__(self, color_mode, dictype, dropout_p=0.30, is_grad=False, separate_step=2):  # color_mode -> 0: gray or 1: RGB
        super(LipNetCNNSeparate, self).__init__()
        self.is_grad = is_grad
        self.dropout_p  = dropout_p
        self.separate_step = separate_step
        self.cnn3ds = []
        for step in range(self.separate_step):
            self.cnn3ds.append(CNN3D_x2(color_mode=color_mode, dropout_p=self.dropout_p, is_grad=self.is_grad).cuda())
        self.cnn3d_seq = nn.Sequential(self.cnn3ds)
        self.gru1  = nn.GRU(96*4*8, 256, 1, bidirectional=True)
        self.gru2  = nn.GRU(512, 256, 1, bidirectional=True)
        #self.gru3 = nn.GRU(512, 256, 1, bidirectional=True)

        fc  = None
        if dictype == 'anno_data_mydic':
            fc = nn.Linear(512, 25+1)
        elif dictype == 'anno_data_romaji':
            fc = nn.Linear(512, 27+1)

        self.FC = fc

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)
        self.dropout3d = nn.Dropout3d(self.dropout_p)
        self._init()

    def _init(self):

        init.kaiming_normal_(self.FC.weight, nonlinearity='sigmoid')
        init.constant_(self.FC.bias, 0)

        for m in (self.gru1, self.gru2):#, self.gru3):
            stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
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


    def forward(self, x):
        # separate x
        xs = []
        sep_num = x.size(2) // self.separate_step
        for step in range(self.separate_step):
            start = step*sep_num
            end = x.size(2) if step == (self.separate_step - 1) \
                else (step+1)*sep_num
            xs.append(x[:, :, start:end, :, :])

        outs = []
        feature_cnns = []
        for idx in range(len(self.cnn3ds)):
            out = self.cnn3ds[idx](xs[idx])
            if self.is_grad:
                feature_cnns.append(out.clone().detach().requires_grad_(True))
            outs.append(out)
        output = torch.cat(outs, dim=2)
        
        # (B, C, T, H, W)->(T, B, C, H, W)
        output = output.permute(2, 0, 1, 3, 4).contiguous() # 軸の順番を変更
        # また、contiguous()はviewにするとき、メモリ上に要素順に並ぶため、エラーを回避できる
        # (B, C, T, H, W)->(T, B, C*H*W)
        output = output.view(output.size(0), output.size(1), -1).contiguous()

        # RNNの重みがメモリ上で非連続にならないように
        # また、パラメータデータポインターをリセットして、より高速なコードパスを使用できるようになっている。
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
        #self.gru3.flatten_parameters()
        output, h = self.gru1(output)
        output = self.dropout(output)
        output, h = self.gru2(output)
        output = self.dropout(output)
        #x, h = self.gru3(x)
        #x = self.dropout(x)
        output = self.FC(output)
        output = output.permute(1, 0, 2).contiguous()
        if self.is_grad:
            return output, feature_cnns
        return output

if __name__ == '__main__':
    inputs = torch.randn(1, 1, 179, 64, 128)
    model = LipNetCNNSeparate(color_mode=0, dictype='anno_data_mydic', separate_step=3)
    print(model)
    # out = model(inputs)
    # print(out.size())
    # print(model.separate_step)