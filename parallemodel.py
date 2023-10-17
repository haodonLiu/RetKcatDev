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
                'lr': 0.01, 'WD': 0.0001,'batch_size': 4
                }
            
        else:self.config=config

        for i in self.config:
            self.__dict__[i]=self.config[i]



        self.seq_ebd=nn.Embedding(36,self.hidden_dim).to('cuda:1')
        self.retnet=RetNet(layers=self.ret_layer,hidden_dim=self.hidden_dim,ffn_size=self.ffn_size,heads=self.heads).to('cuda:1')

        self.mol_ebd_GCN = nn.Embedding(36,self.hidden_dim)
        self.GCN = nn.ModuleList([GCN(self.hidden_dim,self.hidden_dim) for _ in range(self.gcn_layer)])

        self.mol_ebd_EGNNC = nn.Embedding(36,self.hidden_dim).to('cuda:1')
        self.EGNNC = nn.ModuleList([EGNNC(self.hidden_dim,self.hidden_dim) for _ in range(self.egnn_layer)]).to('cuda:1')
        
                
        self.out_L = nn.LeakyReLU()
        self.out = nn.ModuleList([nn.Linear(2,2) for _ in range(self.out_layer)])

        self.trans=nn.Linear(2,1)
        self.average=nn.AvgPool1d(self.hidden_dim)

        self.optimizer= Adam(self.parameters(),lr=self.lr, weight_decay=self.WD)
        
    def gcn(self,x,A):
        
        xs=self.mol_ebd_GCN(x)
        
        for i in range(self.gcn_layer):
            xs=self.GCN[i](xs,A)

        return self.average(xs)

    def egnn(self,x,E):

        xs=self.mol_ebd_EGNNC(x)

        for i in range(self.egnn_layer):
            xs=self.EGNNC[i](xs,E)

        return self.average(xs)

    def ret(self,seq):
        
        x=self.seq_ebd(seq).unsqueeze(0)

        x=self.retnet(x).squeeze(0)
        
        return self.average(x)
        
    def forward(self,inputs):
        
        inputs=iter(inputs+[None])
        seq,mol,A,E=next(inputs)
        output=[]
        
        Xe=self.egnn(mol.to('cuda:1'),E).unsqueeze(0).to('cuda:0')
        Xs=self.ret(seq).T.to('cuda:0')

        for i in inputs:
            Xn=self.gcn(mol,A).unsqueeze(0)
            Xs=torch.cat([torch.sum(Xs*Xn,1),torch.sum(Xs*Xe,1)]).transpose(1,0)
            for i in range(self.out_layer):
                X=self.out_L(self.out[i](Xs))

            output+=[torch.sum(self.trans(X)).unsqueeze(0)]
            
            if i != None :
                seq,mol,A,E=i
                Xe=self.egnn(mol.to('cuda:1'),E).unsqueeze(0).to('cuda:0')
                Xs=self.ret(seq).T.to('cuda:0')
            
        return output

    def Train(self, dataset):
        
        '''return float(loss_total),float(rmse),r2'''
        
        self.train()
        loss_total = 0
        cor_values_list = []
        pre_values_list = []

        for data in dataset[::self.batch_size]:

            batch=[self.dataloader(_) for _ in data]
            correct_values_list.append([i[-1] for i in batch])
            predicted_values_list = self.forward(batch)

            loss = F.mse_loss(correct_values_list, predicted_values_list, reduction='mean')

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            cor_values_list.extend([i.cpu().detach().numpy() for i in predicted_values_list])
            pre_values_list.extend([i.cpu().detach().numpy() for i in correct_values_list])
            loss_total+=loss.cpu().detach().numpy()

        
        correct_values_list=np.array(cor_values_list)
        predicted_values_list=np.array(pre_values_list)
        rmse=torch.sqrt(F.mse_loss(torch.tensor(correct_values_list),torch.tensor(predicted_values_list)))
        r2=r2_score(correct_values_list,predicted_values_list)
        
        del cor_values_list,pre_values_list
        
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
        '''[seq_encode,atom_encode,DA,E,np.log2(value)]'''
        inp=[torch.tensor(data[0],dtype=torch.int).to('cuda:1')]
        inp.append(torch.tensor(data[1],dtype=torch.int).to('cuda:0'))
        inp.append(torch.tensor(data[2],dtype=torch.float).to('cuda:0'))
        inp.append(torch.tensor(data[3],dtype=torch.float).to('cuda:1'))
        inp.append(torch.tensor(data[4],dtype=torch.float).to('cuda:0'))
        return inp
    
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
  