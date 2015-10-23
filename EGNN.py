import torch
import numpy as np
from torch import nn


class EGNN(nn.Module):
    def __init__(self,in_dim,hidden_dim):
        
        super(EGNN, self).__init__()

        self.softM=nn.Softmax(dim=1)
        self.Ai=nn.Parameter(torch.randn(1,in_dim))
        self.Aj=nn.Parameter(torch.randn(1,in_dim))
        self.L=nn.LeakyReLU()
        self.W=nn.Linear(in_dim,hidden_dim)


    def DS(self,E):

        N=E.size()[0]
        e=torch.cat([E[i]*(1/torch.sum(E[i])) for i in range(N)])
        ehat=torch.zeros((N,N))
        S=torch.sum(e,0)
        for i in range(N):
            for j in range(N):
                ehat[i,j]=torch.sum(e[i]*e[j]/S)

        return ehat.to(E.device)

    def alpha(self,X,E):
        '''Refer to PyG'''
        Xh=self.W(X)
        alpha=Xh*self.Ai.sum(1)+Xh*self.Aj.sum(1)
        return self.DS(E@torch.exp(self.L(alpha)))
    
    def forward(self,X,E):
        alpha=self.alpha(X,E)
        return self.softM(self.W(alpha@X)),alpha
    
if __name__ == '__main__':
    E=torch.randn((12,12))
    X=torch.randn((12,10))
    model=EGNN(10,10)
    print(model.forward(X,E))