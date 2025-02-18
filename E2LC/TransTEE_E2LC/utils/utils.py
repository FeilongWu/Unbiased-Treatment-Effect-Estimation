from argparse import Namespace
from typing import Callable, Iterator, Optional, Union

from torch import nn, optim
import csv
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.integrate import romb
from scipy.stats import beta
import pickle


def sigmoid(x):
    return 1 / (1 + np.exp(-x))





def get_patient_outcome(x, v, t, scaling_parameter=10):
    mu = 4 * (t-0.5)**2*np.sin(np.pi/2*t) * 2 * \
             ((sum(v[1]*x) / sum(v[2]*x))**0.5 + 10 * sum(v[0]*x))

    return mu

def evaluate_model(model, data, response):
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


    ####
#    x = torch.tensor([[-1.0251101183452827,-1.6380883167074154,0.6454972243679028,-0.19380063324460373,-0.18085983626508062,-0.5407380704358751,-0.5066473064622846,-1.5116001729444581,-2.5885364776068864,2.264362638853832,-0.5464529087694447,0.6489924321201418,-0.5200358162777989,-0.05496872987736581,-0.0943799931086835,0.09208178150855578,3.589297626148293,-0.34779115729521415,0.18195428377590803,1.4054643799595434,3.186908791034156,3.2741714983769903,-1.5117508904019687,-0.6483370717767875,-1.4085478413792965,1.5400813664319515,1.5811121708640317,1.6866900952680512,1.715233129984588,3.0251257172495616,1.5962127712223342,-0.8787723728682133,-2.078991558347544,0.5289729279084571]
#]).cuda().float()
#    t = torch.from_numpy(treatment_strengths).cuda().float()
#    #t = torch.tensor([0.008, 0.0166, 0.0259, 0.0361, 0.0474, 0.06, 0.0744, 0.0911, 0.1112, 0.1364, 0.1707, 0.2262]
##).cuda().float()
#    pre_y = model.get_predict(x,t)
#    print('pre_y', pre_y.flatten().tolist())
   #########    


    for batch in data:
        x = batch['x'].cuda().float()
        idx = batch['ids'].float().item()
        t = torch.from_numpy(treatment_strengths).cuda().float()
        pre_y = model.get_predict(x,t)
        pred_dose_response = pre_y.flatten().detach().cpu().numpy()
        
        patient = x[0].detach().cpu().numpy()

        true_outcomes = response[idx]
        mise = romb(np.square(true_outcomes - pred_dose_response), dx=step_size)
        mises.append(mise)

    return np.sqrt(np.mean(mises))



def load_data(dataset):
    with open('../data/' + dataset + '_response_curve_calibrate.pickle', 'rb') as file:
        response_data = pickle.load(file)

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
    return x, d, y, ids, response_data

def data_split(x,d,y,ids, test_ratio, num_treatments=1):
    n = len(d)
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

    for i in idx[train_size:]:
        data_te['x'].append(x[i])
        data_te['d'].append(d[i])
        data_te['y'].append(y[i])
        data_te['ids'].append(ids[i])

    return data_tr, data_te


def get_permutations(options, size):
    idx = [0 for i in range(size)]
    r = len(options)
    r1 = r - 1
    last = [r1 for i in range(size)]
    num_combination = r ** size
    combination = []
    last_idx = size - 1
    pointer = last_idx
    for i in range(num_combination):
        if r not in idx:
            combination.append(options[idx].tolist())
            idx[pointer] += 1
        else:
            while r in idx and idx != last:
                pointer = idx.index(r)
                idx[pointer] = 0
                pointer -= 1
                idx[pointer] += 1
            combination.append(options[idx].tolist())
            pointer = last_idx
            idx[pointer] += 1
    return combination

def get_opt_samples(data_tr, density_model, parameters, size, std_w=3):
    errors = {}
    samples = {}
    inv_density = {}
    n = len(parameters)
    cdf = torch.linspace(0,1,size+2)[1:size+1].tolist()
    density_model.eval()
    train_size = 0
    for i in range(n):
        a,b = parameters[i]
        dist = beta(a,b)
        samples[i] = []
        inv_density[i] = []
        for j in cdf:
            t = dist.ppf(j)
            samples[i].append(t)
            inv_density[i].append(1/dist.pdf(t))

        errors[i] = 0
        for idx, batch in enumerate(data_tr):
            x = batch['x'].float()
            bs = x.shape[0]
            for j in range(size):
                propensity = density_model.log_prob(torch.tensor([samples[i][j]]*bs).float()\
                                                    .unsqueeze(-1)-0.5,x)
                errors[i] += np.sum(1 / np.exp(propensity.detach().numpy())) * inv_density[i][j]
    rank_error = []
    ## add samples and weights for uniform distribution
    samples[n] = torch.linspace(0,1,size).tolist()
    inv_density[n] = torch.ones(size).tolist()
    errors[n] = 0
    for idx, batch in enumerate(data_tr):
        x = batch['x'].float()
        bs = x.shape[0]
        train_size += bs
        for j in range(size):
            propensity = density_model.log_prob(torch.tensor([samples[n][j]]*bs).float()\
                                                    .unsqueeze(-1)-0.5,x)
            errors[n] += np.sum(1 / np.exp(propensity.detach().numpy())) * inv_density[n][j]
 
    for i in range(n+1):
        errors[i] -= np.var(samples[i]) * size *  train_size * std_w
        

    
    for i in errors:
        rank_error.append((errors[i], i))
    rank_error = sorted(rank_error)
    idx = rank_error[0][1]
    if idx == len(parameters):
        print('uniform')
    else:
        print(parameters[idx])  
    return torch.tensor(samples[idx]).unsqueeze(-1).float().cuda(), \
           torch.tensor(inv_density[idx]).reshape(1,size).float().cuda()



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


def export_result(out_path, Mise, lr, h_dim, std_w, y_std, cov_dim, dz, t_grid, hidden):
    row = 'lr: ' + str(lr) + '_h_dim: ' + str(h_dim) + '_std_w: ' + str(std_w) + '_y_std: ' + \
          str(y_std) + '_cov_dim: ' + str(cov_dim) + '_dz: ' + str(dz) + '_t_grid: ' + \
          str(t_grid)+ '_hidden: ' + str(hidden)  + ' -- '
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
    elif name == 'sigmoid' or name == 'logistic':
        return nn.Sigmoid()
    else:
        raise Exception("Unknown activation")

def get_iter(data_matrix, batch_size, shuffle=True):
    dataset = Dataset_from_matrix(data_matrix)
    iterator = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return iterator
