import torch
import math
from torch import nn 

class GCN(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.BN = nn.BatchNorm1d(out_features)
        self.L = nn.LeakyReLU()
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input,self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return self.L(self.BN(output + self.bias))
        else:
            return self.L(self.BN(output))

if __name__ == '__main__':
    E=torch.randn((9,9))
    X=torch.randn((9,3))
    model=GCN(3,3,bias=True)
    print(model.forward(X,E))