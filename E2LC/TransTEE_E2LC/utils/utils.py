from argparse import Namespace
from typing import Callable, Iterator, Optional, Union

from torch import nn, optim
import csv
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.integrate import romb


def get_patient_outcome(x, v, t, scaling_parameter=10):
    mu = 4 * (t-0.5)**2*np.sin(np.pi/2*t) * 2 * \
             ((sum(v[1]*x) / sum(v[2]*x))**0.5 + 10 * sum(v[0]*x))

    return mu

def evaluate_model(model, data, v):
    mises = []
    dosage_policy_errors = []
    policy_errors = []
    pred_best = []
    pred_vals = []
    true_best = []
    samples_power_of_two = 6
    num_integration_samples = 2 ** samples_power_of_two + 1
    step_size = 1. / num_integration_samples
    treatment_strengths = np.linspace(np.finfo(float).eps, 1, num_integration_samples)

    for batch in data:
        x = batch['x'].cuda().float()
        t = torch.from_numpy(treatment_strengths).cuda().float()
        pre_y = model.get_predict(x,t)
        pred_dose_response = pre_y.flatten().detach().cpu().numpy()
        
        patient = x[0].detach().cpu().numpy()

        true_outcomes = [get_patient_outcome(patient, v, d) for d in
                                 treatment_strengths]
        mise = romb(np.square(true_outcomes - pred_dose_response), dx=step_size)
        mises.append(mise)

    return np.sqrt(np.mean(mises))



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

def data_split(x,d,y, test_ratio, num_treatments=1):
    n = len(d)
    idx = np.arange(n)
    np.random.shuffle(idx)
    train_size = int(n * (1 - test_ratio))
    propensity = []
    data_tr = {'x':[], 't':[],'d':[],'y':[]}
    data_te = {'x':[], 't':[],'d':[],'y':[]}
    for i in idx[:train_size]:
        data_tr['x'].append(x[i])
        data_tr['d'].append(d[i])
        data_tr['y'].append(y[i])

    for i in idx[train_size:]:
        data_te['x'].append(x[i])
        data_te['d'].append(d[i])
        data_te['y'].append(y[i])

    return data_tr, data_te

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
        return dic


def export_result(out_path, Mise, lr, h_dim, h_inv_eqv_dim, alpha, cov_dim, dz, t_grid):
    row = 'lr: ' + str(lr) + '_h_dim: ' + str(h_dim) + '_h_inv_eqv_dim: ' + str(h_inv_eqv_dim) + '_alpha: ' + \
          str(alpha) + '_cov_dim: ' + str(cov_dim) + '_dz: ' + str(dz) + '_t_grid: ' + str(t_grid)  + ' -- '
    row += 'MISE: (' + str(np.mean(Mise)) + ', ' + str(np.std(Mise)) + ')\n'
    file = open(out_path, 'a')
    file.write(row)
    file.close()


def early_stop(epoch, best_epoch, tol=17):
    if epoch - best_epoch > tol:
        return True
    else:
        return False

    

def get_initialiser(name: str) -> Callable:
    if name == "orthogonal":
        return nn.init.orthogonal_
    elif name == "xavier":
        return nn.init.xavier_uniform_
    elif name == "kaiming":
        return nn.init.kaiming_uniform_
    elif name == "none":
        pass
    else:
        raise Exception("Unknown init method")


def get_optimizer(
    args: Namespace, params: Iterator[nn.Parameter], net: Optional[str] = None
) -> optim.Optimizer:
    weight_decay = args.weight_decay
    lr = args.lr

    optimizer = None
    if args.optimizer == "sgd":
        optimizer = optim.SGD(params, lr=lr, weight_decay=weight_decay)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif args.optimizer == "amsgrad":
        optimizer = optim.Adam(params, lr=lr, amsgrad=True, weight_decay=weight_decay)
    return optimizer


class NoneScheduler:
    def step(self):
        pass


def get_lr_scheduler(
    args: Namespace, optimizer: optim.Optimizer
) -> Union[
    optim.lr_scheduler.ExponentialLR,
    optim.lr_scheduler.CosineAnnealingLR,
    optim.lr_scheduler.CyclicLR,
    NoneScheduler,
]:
    if args.lr_scheduler == "exponential":
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=0)
    elif args.lr_scheduler == "cycle":
        return optim.lr_scheduler.CyclicLR(
            optimizer, 0, max_lr=args.lr, step_size_up=20, cycle_momentum=False
        )
    elif args.lr_scheduler == "none":
        return None


def get_optimizer_scheduler(
    args: Namespace, model: nn.Module
):
    optimizer = get_optimizer(args=args, params=model.parameters())
    lr_scheduler = get_lr_scheduler(args=args, optimizer=optimizer)
    return optimizer, lr_scheduler

def get_initialiser(name='xavier'):
    if name == "orthogonal":
        return nn.init.orthogonal_
    elif name == "xavier":
        return nn.init.xavier_uniform_
    elif name == "kaiming":
        return nn.init.kaiming_uniform_
    elif name == "none":
        pass
    else:
        raise Exception("Unknown init method")


def get_activation(name: str, leaky_relu: Optional[float] = 0.5) -> nn.Module:
    if name == "leaky_relu":
        return nn.LeakyReLU(leaky_relu)
    elif name == "rrelu":
        return nn.RReLU()
    elif name == "relu":
        return nn.ReLU()
    elif name == "elu":
        return nn.ELU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name == "selu":
        return nn.SELU()
    else:
        raise Exception("Unknown activation")

def get_iter(data_matrix, batch_size, shuffle=True):
    dataset = Dataset_from_matrix(data_matrix)
    iterator = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return iterator
