from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class LRR(nn.Module):
    def __init__(self, dim_input, dim_hidden):
        super(LRR, self).__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.B = nn.Linear(dim_input, dim_hidden, bias=False)
        self.w = nn.Linear(dim_hidden, 1, bias=False)
        nn.init.normal_(self.B.weight, mean=0, 
            std=1 / math.sqrt(self.dim_hidden * self.dim_input))
        nn.init.normal_(self.w.weight, mean=0, 
            std=1 / math.sqrt(self.dim_hidden))

    def forward(self, x):
        h = self.B(x)
        y = self.w(h)
        return y


class MLP(nn.Module):
    def __init__(self, dim_input, dim_hidden):
        super(MLP, self).__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.activated = None
        self.down = nn.Linear(dim_input, dim_hidden, bias=True)
        self.up = nn.Linear(dim_hidden, 1, bias=True)
        nn.init.normal_(self.down.weight, mean=0, 
            std=4 / math.sqrt(self.dim_hidden * self.dim_input))
        nn.init.normal_(self.down.bias, mean=0, 
            std=4 / math.sqrt(self.dim_hidden))
        nn.init.normal_(self.up.weight, mean=0, 
            std=4 / math.sqrt(self.dim_hidden))
        nn.init.normal_(self.up.bias, mean=0, 
            std=4 / math.sqrt(self.dim_input))

    def forward(self, x):
        h = self.down(x)
        h = F.relu(h)
        self.activated = h.clone().detach()
        y = self.up(h)
        return y



