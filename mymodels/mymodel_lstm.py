import torch.nn as nn
import torch.nn.init as init
from models.lstm import LSTM


class MyLSTM(nn.Module):
    def __init__(self, T, input_size, hidden_size, num_layers, bidirectional=True, dropout_p=0, is_dropout=False):
        """
            T: This param is frame length
            input_size: This param is input feature size
            hidden_size: This param is hidden layer size
            num_layers: This param is number of LSTM Block
            bidirectional: This param is directional of LSTM
            dropout_p: Dropout rate
        """
        super(MyLSTM, self).__init__()
        dp_p = 0
        if num_layers > 1:
            dp_p = dropout_p
        self.LSTM = LSTM(
            input_size
            ,hidden_size
            ,num_layers
            ,bidirectional=bidirectional
            ,dropout_p=dp_p
        )
        self.dropout = None
        self.is_dropout = is_dropout
        if self.is_dropout:
            self.dropout = nn.Dropout(dropout_p)
        self.FC = nn.Linear(T*hidden_size*2, 5)
        self._init_weight()

    def _init_weight(self):
        init.kaiming_normal_(self.FC.weight, nonlinearity='sigmoid')
        init.constant_(self.FC.bias, 0)
        
    def forward(self, x):
        x, (h,c) = self.LSTM(x)
        x = x.view(x.size(1), -1).contiguous()
        if self.is_dropout:
            x = self.dropout(x)
        x = self.FC(x)
        return x

