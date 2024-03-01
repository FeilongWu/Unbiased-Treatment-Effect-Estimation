import argparse
import os
import numpy as np
import torch.nn.functional as F
import json
import os

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from TransTEE import DA_model
from utils.utils import *
from utils.DisCri import DisCri


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
    parser.add_argument("--alpha", default=1.0, type=float)

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

def pretrain_main(model, args, data, rep, tol=25):
    optimizer, scheduler = get_optimizer_scheduler(args=args, model=model)
    best_loss = np.inf
    model_pth = '../TransTEE_EM/rep' + str(rep) + '.pt'
    for epoch in range(args.max_epochs):
        cum_loss = 0
        for idx, batch in enumerate(data_tr):
            X_mb = Variable(batch['x']).cuda().detach().float()
            T_mb = Variable(torch.ones(X_mb.shape[0],1)).cuda().detach().float()
            D_mb = Variable(batch['t']).cuda().detach().float()
            Y_mb = Variable(batch['y']).cuda().detach().float().unsqueeze(-1)

            optimizer.zero_grad()
            pred_outcome = model(X_mb, T_mb, D_mb)
            if 'tr' in args.model_name:
                set_requires_grad(TargetReg, True)
                tr_optimizer.zero_grad()
                trg = TargetReg(pred_outcome[0].detach())
                if args.p == 1:
                    loss_D = F.mse_loss(trg, T_mb)
                elif args.p == 2:
                    loss_D = neg_guassian_likelihood(trg, T_mb)
                loss_D.backward()
                tr_optimizer.step()

                set_requires_grad(TargetReg, False)
                trg = TargetReg(pred_outcome[0])
                if args.p == 1:
                    loss_D = F.mse_loss(trg, T_mb)
                elif args.p == 2:
                    loss_D = neg_guassian_likelihood(trg, T_mb)
                loss = F.mse_loss(input=pred_outcome[1], target=Y_mb) - args.beta * loss_D
                loss.backward()
                optimizer.step()
                cum_loss += loss.item()
            else:
                loss = F.mse_loss(input=pred_outcome[1], target=Y_mb)
                loss.backward()
                optimizer.step()
            cum_loss += loss.item()
        if scheduler is not None:
            scheduler.step()
                                
        if cum_loss < best_loss:
            best_loss = cum_loss
            best_epoch = epoch
            torch.save(model.state_dict(), model_pth)
        if early_stop(epoch, best_epoch, tol=25):
            break
    model.load_state_dict(torch.load(model_pth))
    return model
    

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
    

def neg_guassian_likelihood(d, u):
    """return: -N(u; mu, var)"""
    B, dim = u.shape[0], 1
    assert (d.shape[1] == dim * 2)
    mu, logvar = d[:, :dim], d[:, dim:]
    return 0.5 * (((u - mu) ** 2) / torch.exp(logvar) + logvar).mean()

if __name__ == "__main__":

    args = init_arg()

    dataset_params = dict()
    dataset_params['num_treatments'] = args.num_treatments
    dataset_params['treatment_selection_bias'] = args.treatment_selection_bias
    dataset_params['dosage_selection_bias'] = args.dosage_selection_bias
    dataset_params['save_dataset'] = args.save_dataset
    dataset_params['validation_fraction'] = args.validation_fraction
    dataset_params['test_fraction'] = args.test_fraction


    dataset = 'synthetic'
    out_path = './TransTEE_' + dataset + '.txt'
    file = open(out_path, 'w')
    file.write('')
    file.close()
    replications = 5
    args.max_epochs = 500
    test_ratio = 0.2
    batch_size = 150
    
    hyperparameters = {'mimic':{'num_dosage_samples':[5], 'h_inv_eqv_dims':[64],\
                                'h_dims':[62], 'alphas':[0.5],\
                                'lrs':[0.001,0.0001],'cov_dims':[80], 't_grids':[5,10,15], 'dzs':[0.9,1.0,1.1]},\
                       'eicu':{'num_dosage_samples':[5], 'h_inv_eqv_dims':[64],\
                                'h_dims':[36], 'alphas':[0.5],\
                                'lrs':[0.001,0.0001],'cov_dims':[100], 't_grids':[5,10,15], 'dzs':[0.9,1.0,1.1]},\
                       'synthetic':{'num_dosage_samples':[5], 'h_inv_eqv_dims':[64],\
                                'h_dims':[52], 'alphas':[0.5],\
                                'lrs':[0.0001],'cov_dims':[60], 't_grids':[8], 'dzs':[0.9]}}[dataset]
                    
    result = {}
    result['in'] = []
    result['out'] = []
    for lr in hyperparameters['lrs']:
        for h_dim in hyperparameters['h_dims']:
            for h_inv_eqv_dim in hyperparameters['h_inv_eqv_dims']: # no use
                for alpha in hyperparameters['alphas']: # no use also no use for num_dosage_samples 
                    for cov_dim in hyperparameters['cov_dims']:
                        for dz in hyperparameters['dzs']:
                            for t_grid in hyperparameters['t_grids']:
                                np.random.seed(3)
                                Mise = []
                                for rep in range(replications):
                                    args.lr = lr
                                    args.h_dim = h_dim
                                    args.h_inv_eqv_dim = h_inv_eqv_dim
                                    args.alpha = alpha
                                    args.cov_dim = cov_dim
                                    args.dz = dz
                                    args.t_grid = t_grid
                            
                                    torch.manual_seed(3)
                                    x,t,y,v = load_data(dataset)
                                    data_tr, data_te = data_split(x,t,y, test_ratio)
                                    data_tr = DataLoader(createDS(data_tr), batch_size=batch_size, shuffle=True)
                                    data_te = DataLoader(createDS(data_te), batch_size=1, shuffle=False)
                                    params = {'num_features': x.shape[1], 'num_treatments': args.num_treatments,
                                'num_dosage_samples': args.num_dosage_samples, 
                                'alpha': args.alpha, 'batch_size': args.batch_size, 'h_dim': args.h_dim,
                                'h_inv_eqv_dim': args.h_inv_eqv_dim, 'cov_dim':args.cov_dim, 'initialiser':args.initialiser,\
                                      's':args.s, 'dz':args.dz, 't_grid':args.t_grid}

                                    if 'tr' in args.model_name:
                                        TargetReg = DisCri(args.h_dim,dim_hidden=50, dim_output=args.p)#args.num_treatments
                                        TargetReg.cuda()
                                        tr_optimizer = torch.optim.SGD(TargetReg.parameters(), lr=0.001, weight_decay=5e-3)


                                    pre_main_path = './rep' + str(rep) + '.pt'

                                    model = DA_model(params)
                                    if os.path.exists(pre_main_path):
                                        model.main_model.load_state_dict(torch.load(pre_main_path))
                                        model.cuda()
                                        print('load pretrained main model from checkpoint')
                                    else:
                                        model.main_model = pretrain_main(model, args, data_tr, rep)
                                        model.cuda()
                                        print('finish pretraining main model')
                                        
                                        

                                    model = pretrain_aux(model, args, data_tr)

                            
                                    
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
                                    mise = evaluate_model(model.main_model, data_te, v)
                                    Mise.append(mise)
                                if len(Mise) == replications:
                                    export_result(out_path, Mise, lr, h_dim, h_inv_eqv_dim, alpha, cov_dim, dz, t_grid)
