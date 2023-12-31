import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from retnet import RetNet
from GCN import GCN
from EGNNC import EGNNC
from sklearn.metrics import r2_score



class RetKcat(nn.Module):
    
    def __init__(self,config=None):
        
        torch.manual_seed(3407) # https://arxiv.org/abs/2109.08203
        super(RetKcat, self).__init__()
        
        if config==None:
            self.config={
                'gcn_layer': 2 , 'egnn_layer':2 , 'out_layer': 3,
                'ret_layer': 2 , 'heads': 4 , 'hidden_dim': 8, 'ffn_size': 16,
                'lr': 0.01, 'WD': 0.0001
                }
        else:self.config=config
        self.device='cuda' if torch.cuda.is_available() else 'cpu'

        for i in self.config:
            self.__dict__[i]=self.config[i]



        self.seq_ebd=nn.Embedding(36,self.hidden_dim)
        self.retnet=RetNet(layers=self.ret_layer,hidden_dim=self.hidden_dim,ffn_size=self.ffn_size,heads=self.heads)

        self.mol_ebd=nn.Embedding(36,self.hidden_dim)
        self.GCN = nn.ModuleList([GCN(self.hidden_dim,self.hidden_dim) for _ in range(self.gcn_layer)])
        self.EGNN = nn.ModuleList([EGNNC(self.hidden_dim,self.hidden_dim) for _ in range(self.egnn_layer)])
        self.GCN_L=nn.LeakyReLU()
                
        self.out_L=nn.LeakyReLU()
        self.out=nn.ModuleList([nn.Linear(2,2) for _ in range(self.out_layer)])

        self.trans=nn.Linear(2,1)
        self.average=nn.AvgPool1d(self.hidden_dim)

        self.optimizer= Adam(self.parameters(),lr=self.lr, weight_decay=self.WD)
        
    def gcn(self,x,A):
        
        xs=self.mol_ebd(x)
        
        for i in range(self.gcn_layer):
            xs=self.GCN_L(self.GCN[i](xs,A))

        return self.average(xs)

    def egnn(self,x,E):

        xs=self.mol_ebd(x)

        for i in range(self.egnn_layer):
            xs=self.EGNN[i](xs,E)

        return self.average(xs)

    def ret(self,seq):
        
        x=self.seq_ebd(seq).unsqueeze(0)

        x=self.retnet(x).squeeze(0)
        
        return self.average(x)
        
    def forward(self,inputs):

        seq,mol,A,E=inputs
        Xn=self.gcn(mol,A).unsqueeze(0)
        Xe=self.egnn(mol,E).unsqueeze(0)
        Xs=self.ret(seq).T

        Xs=torch.cat([torch.sum(Xs*Xn,1),torch.sum(Xs*Xe,1)]).transpose(1,0)

        for i in range(self.out_layer):
            X=self.out_L(self.out[i](Xs))
        
        X=self.trans(X)
        
        return torch.mean(X).unsqueeze(0)

    def Train(self, dataset):
        
        '''return float(loss_total),float(rmse),r2'''
        
        self.train()
        loss_total = 0
        predicted_values_list=[]
        correct_values_list=[]
        
        for data in dataset:

            inp,correct_value=self.dataloader(data)

            predicted_value = self.forward(inp)

            loss = F.mse_loss(predicted_value,correct_value)
            correct_values_list.append(correct_value.cpu().detach().numpy())
            predicted_values_list.append(predicted_value.cpu().detach().numpy())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total+=loss.cpu().detach().numpy()
        
        correct_values_list=np.array(correct_values_list)
        predicted_values_list=np.array(predicted_values_list)
        rmse=torch.sqrt(F.mse_loss(torch.tensor(correct_values_list),torch.tensor(predicted_values_list)))
        r2=r2_score(correct_values_list,predicted_values_list)
        
        return float(loss_total),float(rmse),r2
    
    def Test(self, dataset):
        
        '''return float(MAE),float(rmse),r2'''
        
        self.eval()
        np.random.shuffle(dataset)
        
        SAE = 0
        predicted_values_list=[]
        correct_values_list=[]
        
        for data in dataset :
            inp,correct_value=self.dataloader(data)
            
            predicted_value = self.forward(inp)
            
            SAE += sum(torch.abs(predicted_value-correct_value))
            correct_values_list.append(correct_value.cpu().detach().numpy())
            predicted_values_list.append(predicted_value.cpu().detach().numpy())

        correct_values_list=np.array(correct_values_list)
        predicted_values_list=np.array(predicted_values_list)
        rmse=torch.sqrt(F.mse_loss(torch.tensor(correct_values_list),torch.tensor(predicted_values_list)))
        r2=r2_score(correct_values_list,predicted_values_list)
        MAE = SAE / len(dataset)
        
        return float(MAE),float(rmse),r2

    def dataloader(self,data):
        inp=[torch.tensor(i,dtype=torch.int).to(self.device) for i in data[:2]]
        inp.append(torch.tensor(data[2],dtype=torch.float).to(self.device))
        inp.append(torch.tensor(data[3],dtype=torch.float).to(self.device))
        return inp,torch.tensor(data[4],dtype=torch.float).to(self.device)
    
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
  