import torch
import math
import numpy as np
import os
import csv
import json
from torch.utils.data import DataLoader

from models.dynamic_net import Vcnet, Drnet, TR
from data.data import get_iter
from utils.eval import *

import argparse



def load_data(dataset):
    file = open('../data/' + dataset + '_metadata.json')
    meta = json.load(file)
    file.close()
    v1 = meta['v1']
    v2 = meta['v2']
    v3 = meta['v3']

    x = []
    d = []
    y = []
    file = '../data/' + dataset + '_simulate.csv'
    with open(file) as file1:
        reader = csv.reader(file1, delimiter=',')
        for row in reader:
            #t.append(int(row[0]))
            d.append(float(row[1]))
            y.append(float(row[0]))
            temp = []
            for entry in row[2:]:
                temp.append(float(entry))
            x.append(temp)
    x = np.array(x)
    d = np.array(d)
    y = np.array(y)
    return x, d, y, [v1,v2,v3]



def adjust_learning_rate(optimizer, init_lr, epoch):
    if lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * epoch / num_epoch))
    elif lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = init_lr * (decay ** (epoch // step))
    elif lr_type == 'fixed':
        lr = init_lr
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_checkpoint(state, checkpoint_dir='.'):
    filename = os.path.join(checkpoint_dir, model_name + '_ckpt.pth.tar')
    print('=> Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)

# criterion
def criterion(out, y, alpha=0.5, epsilon=1e-6):
    return ((out[1].squeeze() - y.squeeze())**2).mean() - alpha * torch.log(out[0] + epsilon).mean()

def criterion_TR(out, trg, y, beta=1., epsilon=1e-6):
    # out[1] is Q
    # out[0] is g
    return beta * ((y.squeeze() - trg.squeeze()/(out[0].squeeze() + epsilon) - out[1].squeeze())**2).mean()



def export_result(out_path, Mise, num_unit, lr, alpha1, num_grid1):
    row = 'lr: ' + str(lr) + '_num_unit: ' + str(num_unit) + '_alpha: ' + \
          str(alpha1) + '_num_grid: ' + str(num_grid1) + ' -- '
    row += 'MISE: (' + str(np.mean(Mise)) + ', ' + str(np.std(Mise)) + ')\n'
    file = open(out_path, 'a')
    file.write(row)
    file.close()


def early_stop(epoch, best_epoch, tol=17):
    if epoch - best_epoch > tol:
        return True
    else:
        return False


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train with simulate data')

    # i/o
    parser.add_argument('--data_dir', type=str, default='dataset/simu1/eval/0', help='dir of eval dataset')
    parser.add_argument('--save_dir', type=str, default='logs/simu1/eval', help='dir to save result')

    # training
    parser.add_argument('--n_epochs', type=int, default=8, help='num of epochs to train')

    # print train info
    parser.add_argument('--verbose', type=int, default=100, help='print train info freq')

    # plot adrf
    parser.add_argument('--plt_adrf', type=bool, default=True, help='whether to plot adrf curves. (only run two methods if set true; '
                                                                    'the label of fig is only for drnet and vcnet in a certain order)')

    args = parser.parse_args()

    # optimizer
    lr_type = 'fixed'
    wd = 5e-3
    momentum = 0.9
    # targeted regularization optimizer
    tr_wd = 5e-3

    num_epoch = 500

    # check val loss
    verbose = args.verbose

    load_path = args.data_dir
    save_path = args.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

##    data = pd.read_csv(load_path + '/train.txt', header=None, sep=' ')
##    train_matrix = torch.from_numpy(data.to_numpy()).float()
##    data = pd.read_csv(load_path + '/test.txt', header=None, sep=' ')
##    test_matrix = torch.from_numpy(data.to_numpy()).float()
##    data = pd.read_csv(load_path + '/t_grid.txt', header=None, sep=' ')
##    t_grid = torch.from_numpy(data.to_numpy()).float()
##
##    train_loader = get_iter(train_matrix, batch_size=500, shuffle=True)
##    test_loader = get_iter(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)

    grid = []
    MSE = []

    dataset = 'eicu'
    out_path = './VCNet_' + dataset + '.txt'
    file = open(out_path, 'w')
    file.write('')
    file.close()
    test_ratio = 0.2
    batch_size = 150
    hyperparameters = {'mimic':{'num_units':[60], 'lrs':[ 0.001],\
                                'alphas':[0.0], \
                           'num_grids':[9]},\
                       'eicu':{'num_units':[40], 'lrs':[0.001],\
                                'alphas':[0.3,0.5,0.7], \
                           'num_grids':[9,10,11]},\
                       'ihdp_m50':{'num_units':[35,45,55], 'lrs':[0.001, 0.0001],\
                                'alphas':[0.3,0.5,0.7], \
                           'num_grids':[9,10,11]},\
                       'synthetic_unbiased':{'num_units':[50], 'lrs':[0.001],\
                                'alphas':[0], \
                           'num_grids':[10]}}[dataset]
    replications = 5


    
    # choose from {'Tarnet', 'Tarnet_tr', 'Drnet', 'Drnet_tr', 'Vcnet', 'Vcnet_tr'}
    method_list = [ 'Vcnet']
    model_name = method_list[0]
    for num_unit in hyperparameters['num_units']:
        for lr in hyperparameters['lrs']:
            for alpha1 in hyperparameters['alphas']:
                for num_grid1 in hyperparameters['num_grids']:
                    Mise = []
                    
                    num_grid = num_grid1
                    cfg = [(num_unit, num_unit, 1, 'relu'), (num_unit, 1, 1, 'id')]
                    degree = 2
                    knots = [0.33, 0.66]
            
                    if model_name == 'Vcnet_tr' or model_name == 'Drnet_tr' or model_name == 'Tarnet_tr':
                        isTargetReg = 1
                    else:
                        isTargetReg = 0
                    if isTargetReg:
                        tr_knots = list(np.arange(0.1, 1, 0.1))
                        tr_degree = 2
                        TargetReg = TR(tr_degree, tr_knots)
                        TargetReg._initialize_weights()
                    if model_name == 'Vcnet':
                        init_lr = lr
                        alpha = alpha1
                    elif model_name == 'Vcnet_tr':
                        init_lr = lr
                        alpha = alpha1
                        tr_init_lr = 0.001
                        beta = 1.
                    np.random.seed(3)
                    for rep in range(replications):
                        torch.manual_seed(3)
                        x,t,y,v = load_data(dataset)
                        cfg_density = [(x.shape[1], num_unit, 1, 'relu'), (num_unit, num_unit, 1, 'relu')]
                        data_tr, data_te = data_split(x,t,y, test_ratio)
                        data_tr = DataLoader(createDS(data_tr), batch_size=batch_size, shuffle=True)
                        data_te = DataLoader(createDS(data_te), batch_size=1, shuffle=False)
                        model = Vcnet(cfg_density, num_grid, cfg, degree, knots)
                        model._initialize_weights()
                        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=wd, nesterov=True)

                        if isTargetReg:
                            tr_optimizer = torch.optim.SGD(TargetReg.parameters(), lr=tr_init_lr, weight_decay=tr_wd)
    
                            # x: 2d array, t: 1d vector, y: 1d vector
                        best_loss = np.inf
                        for epoch in range(num_epoch):
                            cum_loss = 0

                            for idx, batch in enumerate(data_tr):
                                x=batch['x'].float()
                                t=batch['t'].float()
                                y=batch['y'].float()

                                if isTargetReg:
                                    optimizer.zero_grad()
                                    out = model.forward(t, x)
                                    trg = TargetReg(t)
                                    loss = criterion(out, y, alpha=alpha) + criterion_TR(out, trg, y, beta=beta)
                                    loss.backward()
                                    optimizer.step()

                                    tr_optimizer.zero_grad()
                                    out = model.forward(t, x)
                                    trg = TargetReg(t)
                                    tr_loss = criterion_TR(out, trg, y, beta=beta)
                                    tr_loss.backward()
                                    tr_optimizer.step()
                                    cum_loss += loss.item() + tr_loss.item()
                                else:
                                    optimizer.zero_grad()
                                    out = model.forward(t, x)
                                    loss = criterion(out, y, alpha=alpha)
                                    loss.backward()
                                    optimizer.step()
                                    cum_loss += loss.item()
                            if cum_loss < best_loss:
                                best_loss = cum_loss
                                best_epoch = epoch
                                torch.save(model.state_dict(), './main_rep' + str(rep) + '.pt')
                            if early_stop(epoch, best_epoch, tol=23):
                                break
                        model.load_state_dict(torch.load('./main_rep' + str(rep) + '.pt'))
                        mise = evaluate_model(model, data_te, v)
                        Mise.append(mise)
                    export_result(out_path, Mise, num_unit, lr, alpha1, num_grid1)



##        if isTargetReg:
##            t_grid_hat, mse = curve(model, test_matrix, t_grid, targetreg=TargetReg)
##        else:
##            t_grid_hat, mse = curve(model, test_matrix, t_grid)
##
##        mse = float(mse)
##        print('current loss: ', float(loss.data))
##        print('current test loss: ', mse)
##        print('-----------------------------------------------------------------')
##
##        save_checkpoint({
##            'model': model_name,
##            'best_test_loss': mse,
##            'model_state_dict': model.state_dict(),
##            'TR_state_dict': TargetReg.state_dict() if isTargetReg else None,
##        }, checkpoint_dir=save_path)
##
##        print('-----------------------------------------------------------------')
##
##        grid.append(t_grid_hat)
##        MSE.append(mse)


##    if args.plt_adrf:
##        import matplotlib.pyplot as plt
##
##        font1 = {'family' : 'Times New Roman',
##        'weight' : 'normal',
##        'size'   : 22,
##        }
##
##        font_legend = {'family' : 'Times New Roman',
##        'weight' : 'normal',
##        'size'   : 22,
##        }
##        plt.figure(figsize=(5, 5))
##
##        c1 = 'gold'
##        c2 = 'red'
##        c3 = 'dodgerblue'
##
##        truth_grid = t_grid[:,t_grid[0,:].argsort()]
##        x = truth_grid[0, :]
##        y = truth_grid[1, :]
##        plt.plot(x, y, marker='', ls='-', label='Truth', linewidth=4, color=c1)
##
##        x = grid[1][0, :]
##        y = grid[1][1, :]
##        plt.scatter(x, y, marker='h', label='Vcnet', alpha=1, zorder=2, color=c2, s=20)
##
##        x = grid[0][0, :]
##        y = grid[0][1, :]
##        plt.scatter(x, y, marker='H', label='Drnet', alpha=1, zorder=3, color=c3, s=20)
##
##        plt.yticks(np.arange(-2.0, 1.1, 0.5), fontsize=0, family='Times New Roman')
##        plt.xticks(np.arange(0, 1.1, 0.2), fontsize=0, family='Times New Roman')
##        plt.grid()
##        plt.legend(prop=font_legend, loc='lower left')
##        plt.xlabel('Treatment', font1)
##        plt.ylabel('Response', font1)
##
##        plt.savefig(save_path + "/Vc_Dr.pdf", bbox_inches='tight')
