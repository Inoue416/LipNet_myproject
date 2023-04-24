import torch.nn as nn
import torch.nn.init as init

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=True, dropout_p=0):
        super(LSTM, self).__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, bidirectional=bidirectional, dropout=dropout_p)
        self.init_weights()

    def init_weights(self):
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

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return out, (h, c)


if __name__ == '__main__':
    import torch
    model = LSTM(
        10, 128, 1, True
    )
    input = torch.randn(5, 3, 10)
    output, (h, c) = model(input)
    print(output.shape)