from argparse import Namespace
from typing import Callable, Iterator, Optional, Union

from torch import nn, optim
import csv
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.integrate import romb
import pickle



def sigmoid(x):
    return 1 / (1 + np.exp(-x))






def evaluate_model(model, data, response ):
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

    #count = 0
    #pre_y1 = np.zeros(65)

 #######
#    x = torch.tensor([[-0.0024028752470471,-0.01579938616009668,-0.01579938616009668,-0.01553670947552708,-0.01579938616009668,-0.01579938616009668,-0.01579938616009668,0.010809761986803761,-0.010020499099565488,-0.015077025277530283,-0.010283175784135089,-0.01509015911175876,0.008892222189445685,-0.006894646553187254,-0.012909942629831085,0.02859297353216566,-0.014591073411076521,-0.0011551609953415014,-0.015457906470156203,-0.012226983249950125,0.02018731962593847,-0.006343025515591094,-0.011938038896923566,0.008366868820306485,0.024652823263621663,0.007841515451167286,0.014933785934546476,-0.010808529153274288,-0.006125003867398327,0.009942928927724084,0.02176337973335607]
#]).cuda().float()
#    t = torch.from_numpy(treatment_strengths).cuda().float()
#    pre_y = model.get_predict(x,t)
#    print('est curve')
#    print('pre_y', pre_y.flatten().tolist())
#    t = torch.tensor([0.0, 0.1111111119389534, 0.2222222238779068, 0.3333333432674408, 0.4444444477558136, 0.5555555820465088, 0.6666666865348816, 0.7777777910232544, 0.8888888955116272, 1.0]
#).cuda().float()
#    pre_y = model.get_predict(x,t)
#    print('sampled')
#    print('pre_y', pre_y.flatten().tolist())
    print('reference')
    print(response[70].tolist())


    
   #########    

    for batch in data:
        x = batch['x'].cuda().float()
        idx = batch['ids'].float().item()
        t = torch.from_numpy(treatment_strengths).cuda().float()
        pre_y = model.get_predict(x,t)
        pred_dose_response = pre_y.flatten().detach().cpu().numpy()

        #pre_y1 += pred_dose_response
        #count += 1
        
        patient = x[0].detach().cpu().numpy()

        true_outcomes = response[idx]
        mise = romb(np.square(true_outcomes - pred_dose_response), dx=step_size)
        mises.append(mise)

    #print('pre_y1',(pre_y1/count).tolist())
    #exit(0)

    
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


def export_result(out_path, Mise, lr, h_dim, h_inv_eqv_dim, alpha, cov_dim):
    row = 'lr: ' + str(lr) + '_h_dim: ' + str(h_dim) + '_h_inv_eqv_dim: ' + str(h_inv_eqv_dim) + '_alpha: ' + \
          str(alpha) + '_cov_dim: ' + str(cov_dim)  + ' -- '
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
