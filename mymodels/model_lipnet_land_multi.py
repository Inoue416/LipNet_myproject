import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import numpy as np

class LipNetLandMulti(torch.nn.Module):
    def __init__(self, T, hidden_size, num_layers, anno_kind='anno_data_romaji', color_mode=0, bidirectional=False, batch_first=True, dropout3d_p=0.30, dropout_p=0.5):
        """
            hidden_size    : LSTMの隠れ層のサイズ
            num_layers     : 何ブロック使うか
            anno_kind      : アノテーションの種類
            color_mode     : 1 -> RGB,  0 -> Gray
            bidirectional  : LSTMが双方向であるか
            batch_first    : LSTMがバッチファーストであるか
            dropout3d_p    : 3DCNN部分のドロップアウトのパラメータ
            dropout_p      : LSTM部分のパラメータ
        """
        super(LipNetLandMulti, self).__init__()
        first_channel = 3 if color_mode else 1
        self.conv1 = nn.Conv3d(first_channel, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))


        self.gru1  = nn.GRU(96*4*8, 256, 1, bidirectional=True)
        self.gru2  = nn.GRU(512, 256, 1, bidirectional=True)
        self.FC_lipnet = None
        if anno_kind == 'anno_data_romaji':
            self.FC_lipnet = nn.Linear(512, 27+1)
        else:
            self.FC_lipnet = nn.Linear(512, 25+1)
        dp_p = 0
        if num_layers > 1:
            dp_p = dropout_p
        self.lstm = nn.LSTM(
            136, hidden_size, num_layers, bidirectional=bidirectional, batch_first=batch_first, dropout=dp_p)
        
        self.FC_land = None
        if bidirectional:
            self.FC_land = nn.Linear(T*2*hidden_size, 6)
        else:
            self.FC_land = nn.Linear(T*hidden_size, 6)

        self.bidirectional = bidirectional
        self.dropout_p  = dropout_p
        self.dropout3d_p = dropout3d_p

        self.relu = nn.ReLU(inplace=True)
        self.dropout_lipnet = nn.Dropout(self.dropout3d_p)
        self.dropout3d = nn.Dropout3d(self.dropout3d_p)
        self.dropout = nn.Dropout(self.dropout_p)
        self._init()

    def _init(self):

        init.kaiming_normal_(self.conv1.weight, nonlinearity='relu') # He初期化 重みの初期化
        init.constant_(self.conv1.bias, 0) # 入力テンソルに値を入れる constant_(テンソル, 埋める値) バイアスの初期化

        #以下上記と同じ
        init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        init.constant_(self.conv2.bias, 0)

        init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        init.constant_(self.conv3.bias, 0)

        init.kaiming_normal_(self.FC_lipnet.weight, nonlinearity='sigmoid')
        init.constant_(self.FC_lipnet.bias, 0)

        init.kaiming_normal_(self.FC_land.weight, nonlinearity='sigmoid')
        init.constant_(self.FC_land.bias, 0)


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

        if self.bidirectional:
            for name, param in self.lstm.named_parameters():
                if 'weight_ih_l' in name:
                    init.xavier_uniform_(param)
                elif 'weight_hh_l' in name:
                    init.orthogonal_(param)
                elif 'weight_ih_r' in name:
                    init.xavier_uniform_(param)
                elif 'weight_hh_r' in name:
                    init.orthogonal_(param)
                elif 'bias' in name:
                    init.constant_(param, 0)
        else:
            for name, param in self.lstm.named_parameters():
                if 'weight_ih' in name:
                    init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    init.orthogonal_(param)
                elif 'bias' in name:
                    init.constant_(param, 0)



    def forward(self, x, x2):

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

        # (B, C, T, H, W)->(T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4).contiguous() # 軸の順番を変更
        # また、contiguous()はviewにするとき、メモリ上に要素順に並ぶため、エラーを回避できる
        # (B, C, T, H, W)->(T, B, C*H*W)
        x = x.view(x.size(0), x.size(1), -1).contiguous()

        # RNNの重みがメモリ上で非連続にならないように
        # また、パラメータデータポインターをリセットして、より高速なコードパスを使用できるようになっている。
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
        x, h = self.gru1(x)
        x = self.dropout_lipnet(x)
        x, h = self.gru2(x)
        x = self.dropout_lipnet(x)

        self.lstm.flatten_parameters()
        x2 = x2.contiguous()
        x2, (hn_lstm, cn_lstm) = self.lstm(x2)
        x2 = self.dropout(x2)
        # x2, (hn_lstm, cn_lstm) = self.lstm2(x2)
        # x2 = self.dropout(x2)
        # x2 = x2.permute(1, 0, 2).contiguous()
        x = self.FC_lipnet(x)
        x = x.permute(1, 0, 2).contiguous()
        x2 = x2.contiguous()
        x2 = x2.view(x2.size(0), -1).contiguous()
        x2 = self.FC_land(x2)
        return x, x2


if __name__ == '__main__':
    model = LipNetLandMulti(1120, 32, 1, color_mode=0)
    print(model)
    input1 = torch.randn(2, 1, 1120, 64, 128)
    input2 = torch.randn(2, 1120, 136)
    out, out2 = model(input1, input2)
    print(out.size())
    print(out2.size())
