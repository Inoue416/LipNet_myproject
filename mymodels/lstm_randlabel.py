import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math

class LstmNet(nn.Module):
    def __init__(self):
        super(LstmNet, self).__init__()
        self.lstm1_1 = nn.LSTM(
            input_size=68*2,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.lstm1_2 = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.lstm2_1 = nn.LSTM(
            input_size=68*2,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.lstm2_2 = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.5)
        self.FC = nn.Linear(256, 5)

        self._init_weight()

    def _init_weight(self):
        init.kaiming_normal_(self.FC.weight, nonlinearity='sigmoid')
        init.constant_(self.FC.bias, 0)
        """for i in range(0, 256*4, 256):
            stdv = math.sqrt(2/(2*4*6+256)) 
            init.uniform_(self.lstm.weight_ih_l0[i: i+256],
                -math.sqrt(4)*stdv, math.sqrt(4)*stdv
            )
            init.orthogonal_(self.lstm.weight_hh_l0[i: i+ 256])
            init.constant_(self.lstm.bias_ih_l0[i: i + 256], 0)
            init.uniform_(self.lstm.weight_ih_l0_reverse[i: i + 256],
                        -math.sqrt(4) * stdv, math.sqrt(4) * stdv)
            init.orthogonal_(self.lstm.weight_hh_l0_reverse[i: i + 256])
            init.constant_(self.lstm.bias_ih_l0_reverse[i: i + 256], 0)"""

        for m in (self.lstm1_1, self.lstm1_2):
            for i in range(0, 256*4, 256):
                stdv = math.sqrt(2/(2*4*6+256)) 
                init.uniform_(m.weight_ih_l0[i: i+256],
                    -math.sqrt(4)*stdv, math.sqrt(4)*stdv
                )
                init.orthogonal_(m.weight_hh_l0[i: i+ 256])
                init.constant_(m.bias_ih_l0[i: i + 256], 0)
                init.uniform_(m.weight_ih_l0_reverse[i: i + 256],
                            -math.sqrt(4) * stdv, math.sqrt(4) * stdv)
                init.orthogonal_(m.weight_hh_l0_reverse[i: i + 256])
                init.constant_(m.bias_ih_l0_reverse[i: i + 256], 0)
        for m in (self.lstm2_1, self.lstm2_2):
            for i in range(0, 256*4, 256):
                stdv = math.sqrt(2/(2*4*6+256)) 
                init.uniform_(m.weight_ih_l0[i: i+256],
                    -math.sqrt(4)*stdv, math.sqrt(4)*stdv
                )
                init.orthogonal_(m.weight_hh_l0[i: i+ 256])
                init.constant_(m.bias_ih_l0[i: i + 256], 0)
                init.uniform_(m.weight_ih_l0_reverse[i: i + 256],
                            -math.sqrt(4) * stdv, math.sqrt(4) * stdv)
                init.orthogonal_(m.weight_hh_l0_reverse[i: i + 256])
                init.constant_(m.bias_ih_l0_reverse[i: i + 256], 0)


    def forward(self, x1):
        self.lstm1_1.flatten_parameters()
        self.lstm1_2.flatten_parameters()
        x1 = x1.contiguous()
        x1 = x1.view(x1.size(0), x1.size(1), -1).contiguous()
        x1, (hn1, cn1) = self.lstm1_1(x1)
        x1 = self.dropout(x1)
        # x1, (hn1, cn1) = self.lstm1_2(x1)
        # x1 = self.dropout(x1)
        """self.lstm2_1.flatten_parameters()
        self.lstm2_2.flatten_parameters()
        x2 = x2.contiguous()
        x2 = x2.view(x2.size(0), x2.size(1), -1).contiguous()
        x2, (hn2, cn2) = self.lstm2_1(x2)
        x2 = self.dropout(x2)
        x2, (hn2, cn2) = self.lstm2_2(x2)
        x = x1 + x2"""
        # x = hn1[-1,:,:]
        # x = self.FC(x)
        x = self.FC(x1)
        return x