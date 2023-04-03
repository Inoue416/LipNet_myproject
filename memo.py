import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import numpy as np

conv1 = nn.Conv2d(3, 32, (2, 2), (2, 2))
pool1 = nn.MaxPool2d((2, 2))

FC = nn.Linear(32, 5)
softmax = nn.Softmax(dim=1)

sample = torch.randn(2, 3, 64, 128)

out = conv1(sample)
out = pool1(out)
print(out.shape)
out = FC(out)
print(out.shape)
out = softmax(out)

print(out.shape)