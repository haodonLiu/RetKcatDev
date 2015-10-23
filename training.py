from model import *
import time
import pickle

with open('dataset.pkl','rb') as data:dataset=pickle.load(data)
for i in dataset:
    if i[1].shape != i[2][0].shape:
        print(i[0].shape,i[2].shape)
        exit()

def split_data(data,ratio):
    np.random.shuffle(data)
    n = len(data)
    n_train = int(n*ratio)
    return data[:n_train],data[n_train:]

train_data,val_data = split_data(dataset[:50],0.9)

config = {
    'gcn_layer': 2 , 'egnn_layer':2 , 'out_layer': 3,
    'ret_layer': 2 , 'heads': 4 , 'hidden_dim': 8, 'ffn_size': 16,
    'lr': 0.001, 'WD': 0.0005
    }

rk = RetKcat(config).to('cuda')

log = 'epoch,train_loss,train_rmse,train_r2,val_loss,val_rmse,val_r2\n'

for i in range(10):
    
    time1 = time.time()
    train_loss,train_rmse,train_r2 = rk.Train(train_data)
    
    time2 = time.time()
    print('epoch',i+1,'time',time2-time1,'train_loss:',train_loss,'train_rmse:',train_rmse,'train_r2:',train_r2)
    test_loss,test_rmse,test_r2 = rk.Test(val_data)
    
    time3 = time.time()
    print('epoch',i+1,'time',time3-time2,'val_loss:',test_loss,'val_rmse:',test_rmse,'val_r2:',test_r2)
    log += f'{i+1},{train_loss},{train_rmse},{train_r2},{test_loss},{test_rmse},{test_r2}\n'

with open('log.csv','wt') as f:f.write(log)

with open(f'epoch{i+1}.modelstate','wb') as f:rk.save_model(f)

with open('config.txt','w') as f:f.write(f'{config}\n{rk.__dict__}')

