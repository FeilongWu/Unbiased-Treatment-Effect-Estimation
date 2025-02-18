import argparse
import os
import numpy as np
import torch.nn.functional as F
import json
import os
import copy

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, WeightedRandomSampler
from TransTEE import DA_model
from utils.utils import *
from utils.DisCri import DisCri
import pickle
#import pyro
from CNF import ConditionalNormalizingFlow



def load_data(dataset):
##    file = open('../data/' + dataset + '_metadata.json')
##    meta = json.load(file)
##    file.close()
##    v1 = meta['v1']
##    v2 = meta['v2']
##    v3 = meta['v3']

    
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
    file = open('../data/' + dataset + '_propensity.pickle','rb')
    ps = pickle.load(file) # list
    return x, d, y, ids, ps


def scale_bias(ps, bias_level):
    ps = np.array(ps) ** bias_level
    ps = ps / sum(ps)
    return ps.tolist()


def data_split(x,d,y,ids, ps, test_ratio, num_treatments=1):
    n = len(d)
    ps_tr = []
    idx = np.arange(n)
    np.random.shuffle(idx)
    train_size = int(n * (1 - test_ratio))
    propensity = []
    data_tr = {'x':[], 't':[],'d':[],'y':[], 'ids':[]}
    data_te = {'x':[], 't':[],'d':[],'y':[], 'ids':[]}
    for i in idx[:train_size]:
        data_tr['x'].append(x[i])
        data_tr['d'].append(d[i])
        data_tr['y'].append(y[i])
        data_tr['ids'].append(ids[i])
        ps_tr.append(ps[i])

    for i in idx[train_size:]:
        data_te['x'].append(x[i])
        data_te['d'].append(d[i])
        data_te['y'].append(y[i])
        data_te['ids'].append(ids[i])

    return data_tr, data_te, ps_tr


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad



def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_treatments", default=1, type=int)
    parser.add_argument("--treatment_selection_bias", default=2.0, type=float)
    parser.add_argument("--dosage_selection_bias", default=2.0, type=float)
    parser.add_argument("--save_dataset", default=True)
    parser.add_argument("--validation_fraction", default=0.1, type=float)
    parser.add_argument("--test_fraction", default=0.2, type=float)
    parser.add_argument("--model_name", default="TransTEE_tr")
    parser.add_argument("--verbose", default=False)
    parser.add_argument("--batch_size", default=150, type=int)
    parser.add_argument("--h_dim", default=48, type=int)
    parser.add_argument("--rep", default=1, type=int)
    parser.add_argument("--h_inv_eqv_dim", default=64, type=int)
    parser.add_argument("--num_dosage_samples", default=5, type=int)
    parser.add_argument("--y_std", default=0.2, type=float)

    #optimizer and scheduler
    parser.add_argument('--beta', type=float, default=0.5, help='tradeoff parameter for advesarial loss')
    parser.add_argument('--p', type=int, default=1, help='dim for outputs of treatments discriminator, 1 for value, 2 for mean and var')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "amsgrad"]
    )
    parser.add_argument(
            "--log_interval",
            type=int,
            default=10,
            help="How many batches to wait before logging training status",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["exponential", "cosine", "cycle", "none"],
    )
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--cov_dim", type=int, default=100)
    parser.add_argument("--s", type=int, default=45) # sample size of counterfactuals
    parser.add_argument("--t_grid", type=int, default=2)
    parser.add_argument("--dz", type=int, default=9)
    parser.add_argument(
        "--initialiser",
        type=str,
        default="xavier",
        choices=["xavier", "orthogonal", "kaiming", "none"],
    )
    return parser.parse_args()

def pretrain_aux(model, args, data_tr, tol=25):
    optimizer, scheduler = get_optimizer_scheduler(args=args, model=model)
    best_loss = np.inf
    model_pth = './saved.pt'
    for epoch in range(args.max_epochs):
        cum_loss = 0
        for idx, batch in enumerate(data_tr):
            X_mb = Variable(batch['x']).cuda().detach().float()
            T_mb = Variable(torch.ones(X_mb.shape[0],1)).cuda().detach().float()
            D_mb = Variable(batch['t']).cuda().detach().float()
            Y_mb = Variable(batch['y']).cuda().detach().float().unsqueeze(-1)
            optimizer.zero_grad()
            loss = model.get_pretrain_aux_loss(X_mb, T_mb, D_mb, Y_mb)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            cum_loss += loss.item()
        if cum_loss < best_loss:
            best_loss = cum_loss
            best_epoch = epoch
            torch.save(model.state_dict(), model_pth)
        if early_stop(epoch, best_epoch, tol=25):
            break
    model.load_state_dict(torch.load(model_pth))
    return model


def CNF_best_loss(model, data_tr, path, args, lr, tol=12):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = np.inf
    for epoch in range(args.max_epochs):
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
    for epoch in range(args.max_epochs):
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

    dataset_params = dict()
    dataset_params['num_treatments'] = args.num_treatments
    dataset_params['treatment_selection_bias'] = args.treatment_selection_bias
    dataset_params['dosage_selection_bias'] = args.dosage_selection_bias
    dataset_params['save_dataset'] = args.save_dataset
    dataset_params['validation_fraction'] = args.validation_fraction
    dataset_params['test_fraction'] = args.test_fraction


    dataset = 'eicu'
    with open('../data/' + dataset + '_response_curve_calibrate.pickle', 'rb') as file:
        response_data = pickle.load(file)
    bias_level = 0.1 # base level = 0
    
    out_path = './TransTEE_' + dataset + '_bias_level_' + str(int(bias_level*100)) + '.txt'
    file = open(out_path, 'w')
    file.write('')
    file.close()
    replications = 5
    args.max_epochs = 500 # use 2 for mimiciii-mv
    # use 5 for mimiciv-mv pretrain 1 for max_epochs
    # 15 for mimiciv-mv10
    # use 1 for mimiciv-seda
    # 35 for mimiciii-seda20
    test_ratio = 0.2
    batch_size = 150
    
    hyperparameters = {'mimic':{'num_dosage_samples':[5], 'h_inv_eqv_dims':[64],'std_ws':[0.1],\
                                'h_dims':[62], 'y_stds':[0.5],'hiddens':[0.9]，\
                                'lrs':[0.001,0.0001],'cov_dims':[80], 't_grids':[5,10,15], 'dzs':[0.9,1.0,1.1]},\
                       'eicu':{'num_dosage_samples':[5], 'h_inv_eqv_dims':[64],'std_ws':[0.1],\
                                'h_dims':[36], 'y_stds':[0.5],'hiddens':[0.9]，\
                                'lrs':[0.001,0.0001],'cov_dims':[100], 't_grids':[5,10,15], 'dzs':[0.9,1.0,1.1]},\
                       'synthetic':{'num_dosage_samples':[5], 'h_inv_eqv_dims':[64],'std_ws':[0.1],\
                                'h_dims':[52], 'y_stds':[0.5],'hiddens':[0.9]，\
                                'lrs':[0.001,0.0001],'cov_dims':[60], 't_grids':[5,10,15], 'dzs':[0.9,1.0,1.1]}}[dataset]
                    
    result = {}
    result['in'] = []
    result['out'] = []
    parameters_set = get_permutations(torch.linspace(1,6,10),2)
    # [[10,1],[10,2],[14,1],[14,2]] for mimiciv-mv and mimiciii-mv
    # and mimiciii-seda20
    # get_permutations(torch.linspace(1,6,6),2) for mimiciii-seda
    # and mimiciii-seda10 

    opt_ts_set = {}
    for lr in hyperparameters['lrs']:
        for h_dim in hyperparameters['h_dims']:
            for t_grid in hyperparameters['t_grids']: # no use
                for y_std in hyperparameters['y_stds']: # no use also no use for num_dosage_samples 
                    for cov_dim in hyperparameters['cov_dims']:
                        for dz in hyperparameters['dzs']:
                            for hidden in hyperparameters['hiddens']:
                                hyper_key = (lr,h_dim,t_grid,cov_dim,dz,hidden)
                                if hyper_key not in opt_ts_set:
                                    opt_ts_set[hyper_key] = []
                                for std_w in hyperparameters['std_ws']:
                                    np.random.seed(3)
                                    Mise = []
                                    for r in range(replications):
                                        args.lr = lr
                                        args.h_dim = h_dim
                                        args.y_std = y_std
                                        args.cov_dim = cov_dim
                                        args.dz = dz
                                        args.t_grid = t_grid
                                        
                                        
                                        torch.manual_seed(3)
                                        x,t,y,ids,ps = load_data(dataset)
                                        ps = scale_bias(ps, bias_level)
                                        data_tr, data_te,ps_tr = data_split(x,t,y,ids, ps,test_ratio)
                                        sampler = WeightedRandomSampler(ps_tr, batch_size, replacement=False if batch_size<len(ps_tr) else True)
                                        data_tr = DataLoader(createDS(data_tr), batch_size=batch_size, sampler=sampler)
                                        data_te = DataLoader(createDS(data_te), batch_size=1, shuffle=False)
                                        density_model_path = './CNF.pt'
                                        dx = x.shape[1]
                                        n_layer = 2
                                        hidden1 = int((t_grid + dx)*hidden)
                                        dz1 = int(cov_dim * dz)
                                        encode = get_layer_size(dx+t_grid, hidden1, n_layer, dz1)
                                        density_model = ConditionalNormalizingFlow(input_dim=1, split_dim=0, \
                                                                               context_dim=x.shape[1], hidden_dim=int(cov_dim/2), \
                                                                               num_layers=2, flow_length=1, \
                                                                               count_bins=5, order='quadratic', \
                                                                               bound=0.5, use_cuda=False)
                                        torch.manual_seed(3)
                                        density_model = train_CNF(density_model, data_tr, density_model_path, args)
                                        torch.manual_seed(3)
                                        ts_optimal, sample_w = get_opt_samples(data_tr, density_model, \
                                                                           parameters_set, t_grid, std_w=std_w)
                                        if r == 0 and ts_optimal.tolist() in opt_ts_set[hyper_key]:
                                            break
                                        else:
                                            opt_ts_set[hyper_key].append(ts_optimal.tolist())
                                        params = {'num_features': x.shape[1], 'num_treatments': args.num_treatments,
                                'num_dosage_samples': args.num_dosage_samples, 
                                'y_std': args.y_std, 'batch_size': args.batch_size, 'h_dim': args.h_dim,
                                'h_inv_eqv_dim': args.h_inv_eqv_dim, 'cov_dim':args.cov_dim, 'initialiser':args.initialiser,\
                                      's':args.s, 'dz':args.dz, 't_grid':args.t_grid, 'encode':encode, 'dz1':dz1}

                                        if 'tr' in args.model_name:
                                            TargetReg = DisCri(args.h_dim,dim_hidden=50, dim_output=args.p)#args.num_treatments
                                            TargetReg.cuda()
                                            tr_optimizer = torch.optim.SGD(TargetReg.parameters(), lr=0.001, weight_decay=5e-3)


                                        pre_main_path = './rep' + str(r) + '.pt'

                                        model = DA_model(params, ts_optimal, sample_w, dataset=dataset, y_std=y_std)
                                        model.main_model.load_state_dict(torch.load(pre_main_path))
                                        model.cuda()
                                        print('load pretrained main model from checkpoint')

                                        try:
                                            model = pretrain_aux(model, args, data_tr)
                                        except UnboundLocalError:
                                            print('local variable **best_epoch** referenced before assignment')
                                            break

                            
                                    
                                        optimizer, scheduler = get_optimizer_scheduler(args=args, model=model)
        


                                        best_loss = np.inf
                                        model_pth = './saved.pt'
        
                                        for epoch in range(args.max_epochs):
                                            cum_loss = 0
                                            for idx, batch in enumerate(data_tr):
                                                X_mb = Variable(batch['x']).cuda().detach().float()
                                                T_mb = Variable(torch.ones(X_mb.shape[0],1)).cuda().detach().float()
                                                D_mb = Variable(batch['t']).cuda().detach().float()
                                                Y_mb = Variable(batch['y']).cuda().detach().float().unsqueeze(-1)

                                                optimizer.zero_grad()
                                                loss = model.get_loss(X_mb, T_mb, D_mb, Y_mb)
                                                loss.backward()
                                                optimizer.step()
                                                optimizer.lr = (args.max_epochs - epoch) / \
                                                           args.max_epochs * lr
                                                if scheduler is not None:
                                                    scheduler.step()
                                                cum_loss += loss.item()
                                            if cum_loss < best_loss:
                                                best_loss = cum_loss
                                                best_epoch = epoch
                                                torch.save(model.state_dict(), model_pth)
                                            if early_stop(epoch, best_epoch, tol=25):
                                                break
                                        model.load_state_dict(torch.load(model_pth))
                                        mise = evaluate_model(model.main_model, data_te, response_data)
                                        Mise.append(mise)
                                    if len(Mise) == replications:
                                        export_result(out_path, Mise, lr, h_dim, std_w, \
                                                      y_std, cov_dim, dz, t_grid, hidden)
