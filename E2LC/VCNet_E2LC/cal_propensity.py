import torch
import math
import numpy as np
import os
import copy
from torch.utils.data import DataLoader,Dataset
from CNF import ConditionalNormalizingFlow
import pickle
import argparse
import csv





def early_stop(epoch, best_epoch, tol=17):
    if epoch - best_epoch > tol:
        return True
    else:
        return False




def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_fraction", default=0.2, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_unit", default=60, type=int)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--epochs", default=500, type=int) # 3 for mimiciii-mv, 30 for mimiciv-seda
    parser.add_argument("--wd", default=5e-3, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--lr_main", default=0.001, type=float)
    parser.add_argument("--lr_DA", default=0.001, type=float)
    parser.add_argument("--num_grid", default=10, type=int)
    parser.add_argument("--mu_d", default=3, type=int)
    parser.add_argument("--t_grid", default=40, type=int) # samples of t
    parser.add_argument("--sz", default=40, type=int) # samples of z
    parser.add_argument("--dz", default=30, type=int) # dim of z
    parser.add_argument("--n_layer", default=3, type=int) # hidden layer
    parser.add_argument("--hidden", default=90, type=int) # 1st hidden layer size
    parser.add_argument("--actv", default='relu', type=str)
    parser.add_argument("--s", default=45, type=int) # sample of y
    parser.add_argument("--y_std", default=0.5, type=float)
    return parser.parse_args()


    
class createDS(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['x'])

    def __getitem__(self, idx):
        dic = {}
        dic['x'] = torch.tensor(self.data['x'][idx])
        dic['t'] = torch.tensor(self.data['d'][idx])
        dic['y'] = torch.tensor(self.data['y'][idx])
        dic['ids'] = torch.tensor(self.data['ids'][idx])
        return dic
    
def load_data(dataset):

    x = []
    d = []
    y = []
    ids = []
    file = '../data/' + dataset + '.csv'
    with open(file) as file1:
        reader = csv.reader(file1, delimiter=',')
        for row in reader:
            #t.append(int(row[0]))
            d.append(float(row[1]))
            y.append(float(row[0]))
            ids.append(float(row[-1]))
            temp = []
            for entry in row[2:-1]:
                temp.append(float(entry))
            x.append(temp)
    x = np.array(x)
    d = np.array(d)
    y = np.array(y)
    ids = np.array(ids)
    return x, d, y, ids


def CNF_best_loss(model, data_tr, path, args, lr, tol=12):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = np.inf
    for epoch in range(args.epochs):
        cum_loss = 0
        for idx, batch in enumerate(data_tr):
            x=batch['x'].float()
            t=batch['t'].unsqueeze(-1).float()
            t -= 0.5
            t += torch.randn_like(t) * 0.1
            t = torch.clip(t, -0.5, 0.5)
            loss = -model.log_prob(t, x).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            cum_loss += loss.item()
        if cum_loss < best_loss:
           best_loss = cum_loss
           best_epoch = epoch
           torch.save(model.state_dict(), path)
        if early_stop(epoch, best_epoch, tol=tol):
           break
    return float(best_loss)

def train_CNF(model, data_tr, path, args, tol=12, lrs = [0.001,0.0003,0.0001,0.00005]):
    # choose best lr
    lr_loss = {}
    for lr in lrs:
        lr_loss[lr] = CNF_best_loss(copy.deepcopy(model), data_tr, path, args, lr)
    loss_lr = []
    for i in lr_loss:
        loss_lr.append((lr_loss[i],i))
    best_lr = sorted(loss_lr)[0][1]
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
    best_loss = np.inf
    for epoch in range(args.epochs):
        cum_loss = 0
        for idx, batch in enumerate(data_tr):
            x=batch['x'].float()
            t=batch['t'].unsqueeze(-1).float()
            t -= 0.5
            t += torch.randn_like(t) * 0.1
            t = torch.clip(t, -0.5, 0.5)
            loss = -model.log_prob(t, x).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            cum_loss += loss.item()
        if cum_loss < best_loss:
           best_loss = cum_loss
           best_epoch = epoch
           torch.save(model.state_dict(), path)
        if early_stop(epoch, best_epoch, tol=tol):
           break
    model.load_state_dict(torch.load(path))
    return model


def cal_propensity(model, data,dataset):
    ps = []
    for idx, batch in enumerate(data):
        x=batch['x'].float()
        t=batch['t'].unsqueeze(-1).float()
        t -= 0.5
        t += torch.randn_like(t) * 0.1
        t = torch.clip(t, -0.5, 0.5)
        pro =  torch.exp(model.log_prob(t, x)).item()
        ps.append(pro)
    file = open('../data/' + dataset + '_propensity.pickle', 'wb')
    pickle.dump(ps,file)
    


        
if __name__ == "__main__":
    np.random.seed(3)
    torch.manual_seed(3)

    dataset = 'mimiciv_coag'
    args = init_arg()
    x,t,y,ids = load_data(dataset)
    dx = x.shape[1]
    data_tr = {'x':x,'d':t,'y':y,'ids':ids}
    data_te = DataLoader(createDS(data_tr), batch_size=1, shuffle=False)
    data_tr = DataLoader(createDS(data_tr), batch_size=150, shuffle=True)
    density_model = ConditionalNormalizingFlow(input_dim=1, split_dim=0, \
                                                context_dim=dx, hidden_dim=int(dx*1.1), \
                                                num_layers=2, flow_length=1, \
                                                count_bins=5, order='quadratic', \
                                               bound=0.5, use_cuda=False)
    density_model_path = './CNF_select_bias.pt'
    density_model = train_CNF(density_model, data_tr, \
                                density_model_path, args)
    cal_propensity(density_model, data_te,dataset)

    
    
