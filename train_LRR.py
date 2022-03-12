import torch 
import numpy as np
from toy_model import LRR
import torch.nn as nn
import torch.nn.functional as F
import copy


'''
hyper-parameters:
'''
n = 1000
m = 100
k = 10
d = 3
lr = 0.1
wd = 0
T = 20


def main():
    
    X = torch.normal(0, 1, size=(n, k))
    Z = torch.normal(0, 1, size=(m, k))
    B0 = torch.normal(0, 1, size=(k, d))
    w0 = torch.normal(0, 1, size=(d, 1))
    y = torch.matmul(X, B0)
    y = torch.matmul(y, w0) + torch.normal(0, 0.1, size=(n, 1))
    P = torch.normal(0, 0.001, size=(k, d))

    model_1 = LRR(dim_input=k, dim_hidden=d)
    model_1.B.weight.data = ((B0 + P).clone().detach()).T

    model_2 = copy.deepcopy(model_1)
    model_2.B.weight.requires_grad = False
    #model.w.weight.data = (w0.clone().detach()).T

    loss_1 = nn.MSELoss()
    loss_2 = nn.MSELoss()
    optimizer_1 = torch.optim.SGD(model_1.parameters(),lr=lr,weight_decay=wd)
    optimizer_2 = torch.optim.SGD(model_2.parameters(),lr=lr,weight_decay=wd)
    for i in range(T):
        y1 = model_1(X)
        l1 = loss_1(y1, y) / 2
        l1.backward()
        optimizer_1.step()
        optimizer_1.zero_grad()
        print(torch.norm(model_1.B.weight.clone().detach()))

        y2 = model_2(X)
        l2 = loss_2(y2, y) / 2
        l2.backward()
        optimizer_2.step()
        optimizer_2.zero_grad()
        print(torch.norm(model_2.B.weight.clone().detach()))
        print('From-scratch Loss: %.3f  Pre-trained Loss: %.3f'%(l1.item(), l2.item()))



if __name__ == '__main__':
    main()