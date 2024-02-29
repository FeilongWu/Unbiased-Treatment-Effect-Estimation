'''
This program runs VAE_EM for synthetic data in a single source.
Author: WU Feilong
'''

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import torch
from sklearn.model_selection import train_test_split
import csv
import numpy as np
from argparse import ArgumentParser
from synthetic_con_VAE_EM import *
import copy
import random
from utils import *
from scipy.stats import sem
import math





def getargs():
    parser = ArgumentParser()
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-opt', choices=['adam', 'rmsprop'], default='adam')
    parser.add_argument('-epochs', type=int, default=250)
    parser.add_argument('-atc', type=str, default='elu')
    parser.add_argument('-xcon', type=int, default=28) # nuclsm of  cont. features
    parser.add_argument('-xdis', type=int, default=7) # nuclsm of disc. features
    parser.add_argument('-d', type=int, default=20) # dim of z
    parser.add_argument('-bs', type=int, default=100) # batch size
    parser.add_argument('-hidden', type=int, default=3) # num of hidden layers
    parser.add_argument('-dl', type=int, default=200) # size of a hidden layer
    parser.add_argument('-wdecay', type=float, default=0.0001) # weight decay
    
    return parser.parse_args()



def early_stop(current, last, tol):
    if (current - last) >= tol:
        return True
    else:
        return False


def runmodel(model, data_tr, data_val, data_te, epochs, opt, lr, save_p,  \
            device, args, ear_stop=25, weight_decay=1e-4, n=100):
    if opt.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt.lower() == 'rmsprop':
        optimizer = torch.optim.RMSProp(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        print('Unknown optimizer!')
        exit(0)
        
    lower_bound_best = -np.inf
    x, y = [], []
    for _, batch in enumerate(data_tr):
        x.append(batch['xcon'])
        y.append(batch['y_fac'].unsqueeze(-1))
    whitener_x = PreWhitener(torch.cat(x), device)
    y = torch.cat(y)
    whitener_y = PreWhitener(y, device, store_mu_std=True).to(device)
    nbatch = math.ceil(n / args.bs)
    for epoch in range(epochs):
        model.train()
        for _, batch in enumerate(data_tr):
            loss = 0
            T = batch['T'].to(device).unsqueeze(-1)
            y = whitener_y(batch['y_fac'].to(device).unsqueeze(-1))
            xcon = whitener_x(batch['xcon'].to(device))
            xdis = batch['xdis'].to(device)
            loss += model.get_aux(xcon, xdis, y, T) /  nbatch
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        model.eval()
        lower_bound = 0
        for _, batch in enumerate(data_val):
            T = batch['T'].to(device).unsqueeze(-1)
            y = whitener_y(batch['y_fac'].to(device).unsqueeze(-1))
            xcon = whitener_x(batch['xcon'].to(device))
            xdis = batch['xdis'].to(device)
            lower_bound += model.lower_bound_aux(xcon, xdis, y, T)
        if lower_bound_best < lower_bound:
            lower_bound_best = lower_bound
            torch.save(model.state_dict(),'./MLP.pt') 
            last_update = epoch
            
        if early_stop(epoch, last_update, ear_stop):
            break
    model.load_state_dict(torch.load('./MLP.pt'))
    lower_bound_best = -np.inf

    for epoch in range(epochs):
        model.train()
        for _, batch in enumerate(data_tr):
            loss = 0
            T = batch['T'].to(device).unsqueeze(-1)
            y = whitener_y(batch['y_fac'].to(device).unsqueeze(-1))
            xcon = whitener_x(batch['xcon'].to(device))
            xdis = batch['xdis'].to(device)
            loss += model.getloss(xcon, xdis, y, T) /  nbatch
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # evaluation
        model.eval()
        lower_bound = 0
        for _, batch in enumerate(data_val):
                T = batch['T'].to(device).unsqueeze(-1)
                y = whitener_y(batch['y_fac'].to(device).unsqueeze(-1))
                xcon = whitener_x(batch['xcon'].to(device))
                xdis = batch['xdis'].to(device)
                lower_bound += model.lower_bound(xcon, xdis, y, T)
        
        if lower_bound_best < lower_bound:
            lower_bound_best = lower_bound
            
            

            # test
            
            
            ATE_te, PEHE_te = get_scores(model, data_te, device, whitener_x, whitener_y)
            ATE_tr, PEHE_tr = get_scores(model, data_tr, device, whitener_x, whitener_y)
            last_update = epoch
            
        if early_stop(epoch, last_update, ear_stop):
            break
    #print('Test ITE:' + str(ITE_te) + ', Test ATE: ' + str(ATE_te) + \
    #      ', Test PEHE: ' + str(PEHE_te))

    return ATE_te, PEHE_te, ATE_tr, PEHE_tr
        


def single_dataset(data, replication, save_p, args, \
                   test_ratio, val_ratio, n=100, alpha=1.0):
    # trainDL: batch = {1T:{}, 2T:{},...}
    # validDL: batch = {T:{}, x:{}, ...} contains all data combined
    # testDL: a dic, key = source idx, val = {T:{}, x:{}}
    ATE_ood, PEHE_ood, ATE_within, PEHE_within = [], [], [], []
    for i in range(replication):
        
        if dataset_name == 'IHDP':
            data = read_data_IHDP(path + str(i+1) + '.csv')
        torch.manual_seed(3) # control seed
        data_tr, data_te = train_test_split(data, test_size=test_ratio)
        data_tr, data_val = train_test_split(data_tr, test_size=val_ratio)
        if dataset_name == 'IHDP':
            trainDS = createDS_IHDP(data_tr, args.xdis, args.xcon)
            validDS = createDS_IHDP(data_val, args.xdis, args.xcon)
            trainDL = DataLoader(trainDS, batch_size = args.bs, shuffle=True)
            validDL = DataLoader(validDS, batch_size=50)
            testDL = DataLoader(createDS_IHDP(data_te, args.xdis, args.xcon), batch_size=300)
        else:
            trainDS = createDS(data_tr, args.xdis, args.xcon)
            validDS = createDS(data_val, args.xdis, args.xcon)
            trainDL = DataLoader(trainDS, batch_size = args.bs, shuffle=True)
            validDL = DataLoader(validDS, batch_size=50)
            testDL = DataLoader(createDS(data_te, args.xdis, args.xcon), batch_size=300)
        
        model = VAE_EM(args.xcon, args.xdis, args.d, device, h=args.hidden, dl=args.dl, alpha=alpha).to(device)
        ate_ood, pehe_ood, ate_within, pehe_within = runmodel(model, trainDL, validDL, testDL, \
                                      args.epochs, args.opt, lrate ,save_p,\
                                           device, args, weight_decay=args.wdecay, n=n)
        ATE_ood.append(ate_ood)
        PEHE_ood.append(pehe_ood)
        ATE_within.append(ate_within)
        PEHE_within.append(pehe_within)
    return ATE_ood, PEHE_ood, ATE_within, PEHE_within

    

if __name__ == '__main__':
    # control randomness
    random.seed(3)
    np.random.seed(3)
    ######


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = getargs()
    
    dataset_name = 'synthetic'
    if dataset_name == 'eICU':
        replication = 5
        args.xcon = 28
        args.xdis = 7
        lr = 0.001
        layers = [2]
        hidden_size = [210]
        dimz = [29]
        wdecay = [1e-04]
        n = 1824
        path = '../data/eICU/sourcecombined.csv'
        test_ratio = 0.15
        val_ratio = min(100, n*(1-test_ratio)*0.3)
        val_ratio = val_ratio / (n*(1-test_ratio))
        alphas = [ 1.2]

    elif dataset_name == 'synthetic':
        replication = 5
        args.xcon = 10
        args.xdis = 3
        lr = 0.001
        layers = [2,3,4]
        hidden_size = [110]
        dimz = [6]
        wdecay = [1e-04]
        n=1000
        path = '../data/synthetic/sourcesynthetic_combined.csv'
        test_ratio = 0.15
        val_ratio = min(100, n*(1-test_ratio)*0.3)
        val_ratio = val_ratio / (n*(1-test_ratio))
        alphas = [ 0.8, 1.0, 1.2, 1.8]

    elif dataset_name == 'IHDP':
        replication = 10
        args.xcon = 6
        args.xdis = 19
        lr = 0.001
        layers = [4]
        hidden_size = [200]
        dimz = [25]
        wdecay = [1e-04]
        n=747
        path = '../data/IHDP/ihdp_npci_'
        test_ratio = 0.15
        val_ratio = 0.3
        alphas = [ 1.0, 1.2, 1.8]


    elif dataset_name == 'high_bias':
        replication = 5
        args.xcon = 10
        args.xdis = 3
        lrs = [0.001]
        layers = [2,3,4]
        hidden_size = [80,110,140]
        dimz = [6,10,14]
        wdecay = [1e-04]
        n=500
        path = '../data/high_bias/source8.csv'
        test_ratio = 0.15
        val_ratio = min(100, n*(1-test_ratio)*0.3)
        val_ratio = val_ratio / (n*(1-test_ratio))

        

    out_path = './CEMVAE_alpha_' + dataset_name + '.txt'
    file = open(out_path, 'w')
    file.write('')
    file.close()

    if dataset_name == 'IHDP':
        data = None
    else:
        data = read_data(path)
    lrate = 0.001
        
    for alpha in alphas:
        for num_layer in layers:
            for hsize in hidden_size:
                for dz in dimz:
                    for wd in wdecay:
                        args.wdecay = wd
                        args.d = dz
                        args.lr = lrate
                        args.hidden = num_layer
                        args.dl = hsize
                        
                        save_p = ''
                        
                        try:
                            ATE_ood, PEHE_ood, ATE_within, PEHE_within  = single_dataset(data, replication,  \
                                        save_p, args, test_ratio, val_ratio, n=n, alpha=alpha)
                        except ValueError:
                            continue
                        row = ''
                        row = 'single'+str(n)+'_alpha'+str(alpha)+'_h'+str(num_layer)+\
                                          '_dl'+str(hsize)+'_wd'+str(wd)+'_dz'+str(dz)
                        if dataset_name == 'IHDP':
                            row += 'PEHE-' + str((np.mean(PEHE_ood), sem(PEHE_ood), 'ood', np.mean(PEHE_within), sem(PEHE_within), 'within')) + ' ' + 'ATE-' + \
                                str((np.mean(ATE_ood), sem(ATE_ood), 'ood', np.mean(ATE_within), sem(ATE_within), 'within'))
                        else:
                            row += 'PEHE-' + str((np.mean(PEHE_ood), np.std(PEHE_ood), 'ood', np.mean(PEHE_within), np.std(PEHE_within), 'within')) + ' ' + 'ATE-' + \
                                str((np.mean(ATE_ood), np.std(ATE_ood), 'ood', np.mean(ATE_within), np.std(ATE_within), 'within'))
                        file = open(out_path, 'a')
                        file.write(row + '\n')
                        file.close()
    
    
            
