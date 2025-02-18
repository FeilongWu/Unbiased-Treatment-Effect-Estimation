import torch
import numpy as np
import os
from utils.log_helper import save_obj, load_obj
from torch.utils.data import Dataset




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
    

def sigmod(x):
    return 1. / (1. + torch.exp(-1. * x))

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))

def derivation_sigmoid(x):
    return 1 / (x * (1 - x) + 1e-8)

def load_data(args, name):
    train_file = os.path.join(args.data_dir, name)
    load_dir = args.data_dir

    if not os.path.exists(train_file + '.pkl'):
        print('error: there exist no file-{}'.format(train_file))
        # exit()
    return load_obj(load_dir, name)

def load_train(args):
    return load_data(args, 'train')

def load_test(args):
    return load_data(args, 'test')

def load_eval(args):
    return load_data(args, 'eval')

def save_data(args, data, name):
    path = os.path.join(args.data_dir, name + '.pkl')
    save_dir = args.data_dir
    if os.path.exists(path):
        print('there already exists file-{}, saving data will be ignored'.format(path))
        return
    else:
        save_obj(data, save_dir, name)

def save_train(args, data):
    return save_data(args, data, 'train')

def save_eval(args, data):
    return save_data(args, data, 'eval')

def save_test(args, data):
    return save_data(args, data, 'test')
