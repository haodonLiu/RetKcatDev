from model import *
import time
import pickle

with open('dataset.pkl','rb') as data:dataset=pickle.load(data)

def split_data(data,ratio):
    np.random.shuffle(data)
    n = len(data)
    n_train = int(n*ratio)
    return data[:n_train],data[n_train:]

train_data,val_data = split_data(dataset,0.9)

config = {
    'gcn_layer': 2 , 'egnn_layer':2 , 'out_layer': 3,
    'ret_layer': 3 , 'heads': 4 , 'hidden_dim': 8, 'ffn_size': 16,
    'lr': 0.001, 'WD': 0.0005
    }

rk = RetKcat(config).to('cuda')

log = 'epoch,train_loss,train_rmse,train_r2,val_loss,val_rmse,val_r2\n'

for i in range(10):
    
    time1 = time.time()
    train_loss,train_rmse,train_r2 = rk.Train(train_data)
    
    time2 = time.time()
    print('epoch',i+1,'time',round(time2-time1,4),'train_loss:',round(train_loss,4),'train_rmse:',round(train_rmse,4),'train_r2:',round(train_r2,4))
    test_loss,test_rmse,test_r2 = rk.Test(val_data)
    
    time3 = time.time()
    print('epoch',i+1,'time',round(time3-time2,4),'val_loss:',round(test_loss,4),'val_rmse:',round(test_rmse,4),'val_r2:',round(test_r2,4))
    log += f'{i+1},{train_loss},{train_rmse},{train_r2},{test_loss},{test_rmse},{test_r2}\n'
    with open(f'modelstate/epoch{i+1}.modelstate','wb') as f:rk.save_model(f)

    with open('modelstate/log.csv','wt') as f:f.write(log)

with open('modelstate/config.txt','w') as f:f.write(f'{config}\n{rk.__dict__}')

