
import torch 
import numpy as np
from toy_model import MLP
import torch.nn as nn
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt


'''
hyper-parameters:
'''
n = 1000
m = 100
k = 50
d = 100
lr = 0.1
wd = 0.1
T = 1000

def spst(x):
    return 1 - np.count_nonzero(x) / x.size


def main():
    
    X = torch.normal(10, 1, size=(n, k))
    #Z = torch.normal(0, 1, size=(m, k))

    
    model_0 = MLP(dim_input=k, dim_hidden=d)
    with torch.no_grad():
        y = torch.normal(0, 1e-1, size=(n, 1)) + model_0(X)
        #y = 0 * y
    h0 = model_0.activated.numpy()
    P = torch.normal(0, 1e-4, size=(d, k)) / np.sqrt(d * k)
    q = torch.normal(0, 1e-4, size=(1, d)) / np.sqrt(d)
    
    model_1 = MLP(dim_input=k, dim_hidden=d)
    model_2 = copy.deepcopy(model_1)
    model_2.down.weight.data = model_0.down.weight.data.clone().detach()+P
    model_2.down.bias.data = model_0.down.bias.data.clone().detach()+q

    loss_1 = nn.MSELoss()
    loss_2 = nn.MSELoss()
    optimizer_1 = torch.optim.SGD(model_1.parameters(),lr=lr,weight_decay=wd)
    optimizer_2 = torch.optim.SGD(model_2.parameters(),lr=lr,weight_decay=wd)

    dt = {}
    dt['it'] = []
    dt['ft'] = []
    dt['pt'] = []
    dt['or'] = []

    for i in range(T):
        y1 = model_1(X)
        l1 = loss_1(y1, y) / 2
        l1.backward()
        optimizer_1.step()
        optimizer_1.zero_grad()
        h1 = model_1.activated.numpy()
        #print(torch.norm(model_1.B.weight.clone().detach()))

        y2 = model_2(X)
        l2 = loss_2(y2, y) / 2
        l2.backward()
        optimizer_2.step()
        optimizer_2.zero_grad()
        h2 = model_2.activated.numpy()
        #print(torch.norm(model_2.B.weight.clone().detach()))

        dt['it'].append(i)
        dt['or'].append(spst(h0))
        dt['ft'].append(spst(h1))
        dt['pt'].append(spst(h2))

        if i % 100 == 0:
            print('From-scratch Sparsity: %.3f  Pre-trained Sparsity: %.3f  True Sparsity: %.3f'%(spst(h1), spst(h2), spst(h0)))
            print('From-scratch Loss: %.3f  Pre-trained Loss: %.3f'%(l1.item(), l2.item()))

    L = len(dt['it'])
    plt.plot(dt['it'], dt['or'], label='Original', color='black')
    plt.plot(dt['it'], dt['ft'], label='Fine-tuned', color='red')
    plt.plot(dt['it'], dt['pt'], label='Pre-trained', color='blue')
    plt.xlim(0, 20)
    plt.ylim(-0.1, 1.1)
    plt.xlabel('iteration')
    plt.ylabel('sparsity')
    plt.legend()

    plt.savefig('./fig/sparsity.png', dpi=600)

    
if __name__ == '__main__':
    main()