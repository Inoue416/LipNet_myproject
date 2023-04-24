from models.tcn import TemporalConvNet
import torch.nn as nn
import torch.nn.init as init


class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, num_outputs, T, dropout_p):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(
            num_inputs=num_inputs,
            num_channels=num_channels,
            dropout=dropout_p
        )
        self.FC = nn.Linear(num_channels[-1]*T, num_outputs)
        self._init_weight()

    def _init_weight(self):
        init.kaiming_normal_(self.FC.weight, nonlinearity='sigmoid')
        init.constant_(self.FC.bias, 0)
        
    def forward(self, x):
        x = self.tcn(x)
        x = x.view(x.size(0), -1).contiguous()
        x = self.FC(x)
        return x

if __name__ == "__main__":
    import torch
    from datasets import dataset
    mydataset = dataset.MyDataset(
        "data", "train_path.txt", 1112,
        [1, 1]
        ,[1, 1]
        ,3
    )
    inputs = mydataset.__getitem__(0).get("rand")
    inputs = inputs.unsqueeze(0)
    print(inputs.shape)
    # inputs = inputs.view(inputs.size(0), inputs.size(2), inputs.size(1)).contiguous()
    # inputs = inputs.unsqueeze(0)
    # exit()
    model = TCN(
        num_inputs=inputs.size(1),
        num_channels=[32, 64, 128, 256],
        num_outputs=5,
        T=1112,
        dropout_p=0.2
    )
    out = model(inputs)
    print(out.shape)
    print(out)
    exit()
    softmax = nn.Softmax()
    print(torch.sum(softmax(out)))