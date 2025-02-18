import torch
import math
import numpy as np
import os
import copy
from torch.utils.data import DataLoader
from utils import *
from DA_model import main_model, DA_model
import pyro
from CNF import ConditionalNormalizingFlow


import argparse


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_fraction", default=0.2, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_unit", default=60, type=int)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--epochs", default=500, type=int)
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


    
def pretrain(model, data, args, tol=25):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_main, \
                                momentum=args.momentum, weight_decay=args.wd, \
                                nesterov=True)
    best_loss = np.inf
    for epoch in range(args.epochs):
        cum_loss = 0
        for idx, batch in enumerate(data):
            x=batch['x'].float()
            t=batch['t'].float()
            y=batch['y'].float()
            loss, loss1 = model.get_loss(x,t,y,requires_sample=0)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            cum_loss += loss1
        if best_loss > cum_loss:
            best_loss = cum_loss
            best_epoch = epoch
            torch.save(model.state_dict(), './main_model.pt')
        if early_stop(epoch, best_epoch, tol=tol):
            break
    model.load_state_dict(torch.load('./main_model.pt'))
    return model



def pretrain_aux(model, init_lr, args, data_tr):
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    best_loss = np.inf
    for epoch in range(args.epochs): #args.epochs
        cum_loss = 0

        for idx, batch in enumerate(data_tr):
             x=batch['x'].float()
             t=batch['t'].float()
             y=batch['y'].float()

             loss = model.get_pretrain_aux_loss(x,t,y)
             loss.backward()
             optimizer.step()
             optimizer.zero_grad()
             cum_loss += loss.item()
        if cum_loss < best_loss:
            best_loss = cum_loss
            best_epoch = epoch
            torch.save(model.state_dict(), './saved.pt')
        if early_stop(epoch, best_epoch, tol=10):
            break
    model.load_state_dict(torch.load('./saved.pt'))
    return model

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




def get_layer_size(dx, hidden, n_layer, dz):
    size = [(dx, hidden)]
    interval = (hidden - dz) / (n_layer+1)
    for i in range(n_layer):
        last_out = size[-1][1]
        new_in = int(last_out - interval)
        size.append((last_out, new_in))
    size.append((new_in, dz))
    return size
        




    
if __name__ == "__main__":

    args = init_arg()


    #num_epoch = args.epochs


    MSE = []

    dataset = 'mimiciii_seda'
    out_path = './DA_ps_' + dataset + '.txt'
    file = open(out_path, 'w')
    file.write('')
    file.close()
    test_ratio = 0.2
    batch_size = 150              
    hyperparameters = {'mimiciii_mv':{'num_units':[44], 'lrs':[(0.0003,0.0003)],\
                                'alphas':[0.5], 'num_grids':[11],\
                                'n_layers':[2], 'hiddens':[1.1], 't_grids':[12],\
                                'dzs':[1.1], 'std_ws':[-100], 'y_stds':[0.005]},\
                       'mimiciv_mv':{'num_units':[34], 'lrs':[(0.00005,0.00005)],\
                                'alphas':[0.5], 'num_grids':[11],\
                                'n_layers':[2], 'hiddens':[0.9], 't_grids':[12],\
                                'dzs':[1.05], 'std_ws':[100], 'y_stds':[0.005]},
                       'mimiciv_seda':{'num_units':[32], 'lrs':[(0.0003,0.0003)],\
                                'alphas':[0.5], 'num_grids':[11],\
                                'n_layers':[2], 'hiddens':[1.1], 't_grids':[8],\
                                'dzs':[1.1], 'std_ws':[100],'y_stds':[0.005]},\
                       'mimiciii_seda':{'num_units':[46], 'lrs':[(0.00005,0.00005)],\
                                'alphas':[0.5], 'num_grids':[9],\
                                'n_layers':[2], 'hiddens':[1.0], 't_grids':[12],\
                                'dzs':[1.1], 'std_ws':[100], 'y_stds':[0.005]},\
                       'mimiciv_coag':{'num_units':[38], 'lrs':[(0.00005,0.00005)],\
                                'alphas':[0.5], 'num_grids':[10],\
                                'n_layers':[2], 'hiddens':[0.9], 't_grids':[8],\
                                'dzs':[1.1], 'std_ws':[100], 'y_stds':[0.005]}}[dataset]
    replications = 1
    parameters_set = get_permutations(torch.linspace(1,6,10),2)

    opt_ts_set = {}
    for idx, num_unit in enumerate(hyperparameters['num_units']):
        for lr_main, lr_DA in hyperparameters['lrs']:
            for hidden in hyperparameters['hiddens']:
                for num_grid1 in hyperparameters['num_grids']:
                    for n_layer in hyperparameters['n_layers']:
                        for t_grid in hyperparameters['t_grids']:
                            for dz in hyperparameters['dzs']:
                                for y_std in hyperparameters['y_stds']:
                                    hyper_key = (num_unit,lr_main,lr_DA,hidden,num_grid1,n_layer,t_grid,dz,y_std)
                                    if hyper_key not in opt_ts_set:
                                        opt_ts_set[hyper_key] = []
                                    for std_w in hyperparameters['std_ws']:
                                        Mise = []
                                        args.lr_main = lr_main
                                        num_grid = num_grid1
                                        cfg = [(num_unit, num_unit, 1, 'relu'), (num_unit, 1, 1, 'id')]
                                        degree = 2
                                        knots = [0.33, 0.66]

                    
            
                                        init_lr = lr_DA
                                        np.random.seed(3)
                                        
                                        for rep in range(replications):
                                            x,t,y,ids,response_data = load_data(dataset)
                                            dx = x.shape[1]
                                            hidden1 = int((t_grid + dx)*hidden)
                                            dz1 = int(num_unit * dz)
                                            encode = get_layer_size(dx+t_grid, hidden1, n_layer, dz1)
                                            cfg_aux = [(dz1, dz1, 1, 'relu'), (dz1, 1, 1, 'id')]
                                            cfg_density = [(dx, num_unit, 1, 'relu'), (num_unit, num_unit, 1, 'relu')]
                                            data_tr, data_te = data_split(x,t,y, ids, test_ratio)
                                        

                                            data_tr = DataLoader(createDS(data_tr), batch_size=batch_size, shuffle=True)
                                            data_te = DataLoader(createDS(data_te), batch_size=1, shuffle=False)
                                            density_model_path = './CNF.pt'
                                            density_model = ConditionalNormalizingFlow(input_dim=1, split_dim=0, \
                                                                               context_dim=dx, hidden_dim=num_unit, \
                                                                               num_layers=2, flow_length=1, \
                                                                               count_bins=5, order='quadratic', \
                                                                               bound=0.5, use_cuda=False)
                                            density_model = train_CNF(density_model, data_tr, \
                                                                      density_model_path, args)
                                            ts_optimal, sample_w = get_opt_samples(data_tr, density_model, \
                                                                           parameters_set, t_grid, std_w=std_w)
                                            exit(0)
                                            if rep == 0 and ts_optimal.tolist() in opt_ts_set[hyper_key]:
                                                break
                                            else:
                                                opt_ts_set[hyper_key].append(ts_optimal.tolist())
                                            torch.manual_seed(3)
                                        
                                            main_estimator = main_model(cfg_density, num_grid, cfg, degree, knots,\
                                                         t_grid=t_grid, s=args.s, ts=ts_optimal, dataset=dataset,\
                                                                        y_std=y_std)
                                            main_estimator._initialize_weights()
                                            pre_main_path = './main_rep' + str(rep) + '.pt'

                                            model = DA_model(cfg_density, num_grid, cfg, cfg_aux, degree, knots,\
                                                     dx, encode, ts_optimal, sample_w, act='relu', t_grid=t_grid, \
                                                     s=args.s, dataset=dataset,y_std=y_std)
                                            model._initialize_weights()
                                            pre_main_dict = torch.load(pre_main_path)
                                            for density_param in [ "density_estimator_head.weight", "density_estimator_head.bias"]:
                                                if density_param in pre_main_dict:
                                                    del pre_main_dict[density_param]
                                            model.main_model.load_state_dict(pre_main_dict)
                                            try:
                                                model = pretrain_aux(model, init_lr, args, data_tr)
                                                print('finish pretrain aux')
                                            except UnboundLocalError:
                                                print('local variable **best_epoch** referenced before assignment')
                                                break
                                            optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)

                                # x: 2d array, t: 1d vector, y: 1d vector
                                            #try:
                                            best_loss = np.inf
                                            for epoch in range(args.epochs):
                                                cum_loss = 0

                                                for idx, batch in enumerate(data_tr):
                                                    x=batch['x'].float()
                                                    t=batch['t'].float()
                                                    y=batch['y'].float()

                                                    loss = model.get_loss(x,t,y)
                                                    loss.backward()
                                                    optimizer.step()
                                                    optimizer.zero_grad()
                                                    optimizer.lr = (args.epochs - epoch) / args.epochs\
                                                                   * init_lr
                                                    cum_loss += loss.item()
                                                if cum_loss < best_loss:
                                                    best_loss = cum_loss
                                                    best_epoch = epoch
                                                    torch.save(model.state_dict(), './saved.pt')
                                                if early_stop(epoch, best_epoch, tol=25):
                                                    break
                                            model.load_state_dict(torch.load('./saved.pt'))
                                            mise = evaluate_model(model, data_te, response_data)
                                        #print(best_epoch)
                                        #print(('mise',mise))
                                        #exit(0)
                                            Mise.append(mise)
                                            #except ValueError:
                                                #print('Error')
                                                #break
                                        if len(Mise) == replications:
                                            export_result(out_path, Mise, num_unit, lr_main, lr_DA, hidden, num_grid1, t_grid,\
                                                  n_layer, dz, std_w, y_std)

