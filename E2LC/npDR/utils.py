import csv
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from torch import nn
from scipy.integrate import romb
from scipy.stats import beta
import pickle


def sigmoid(x):
    return 1 / (1 + np.exp(-x))



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






def evaluate_model(curve, data_te, response):
    mises = []
    samples_power_of_two = 6
    num_integration_samples = 2 ** samples_power_of_two + 1
    step_size = 1. / num_integration_samples


    for ids in data_te['ids']:
        true_outcomes = response[ids]
        mise = romb(np.square(true_outcomes - curve), dx=step_size)
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

def data_split(x,t,y,ids, test_ratio, num_treatments=1):
    n = len(t)
    idx = np.arange(n)
    np.random.shuffle(idx)
    train_size = int(n * (1 - test_ratio))
    propensity = []
    data_tr = {'tx':[],'y':[], 'ids':[]}
    data_te = {'tx':[],'y':[], 'ids':[]}
    for i in idx[:train_size]:
        temp = [t[i]]
        temp.extend(x[i].tolist())
        data_tr['tx'].append(temp)
        data_tr['y'].append(y[i])
        data_tr['ids'].append(ids[i])

    for i in idx[train_size:]:
        temp = [t[i]]
        temp.extend(x[i].tolist())
        data_te['tx'].append(temp)
        data_te['y'].append(y[i])
        data_te['ids'].append(ids[i])

    for key in data_tr:
        data_tr[key] = np.array(data_tr[key])
        data_te[key] = np.array(data_te[key])
        

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


def export_result(Mise, out_path, lr, num_unit, layer):
    row = 'lr: ' + str(lr) + '_num_unit: ' + str(num_unit) + '_layer: ' + str(layer) \
          + ' -- '
    row += 'MISE: (' + str(np.mean(Mise)) + ', ' + str(np.std(Mise)) + ')\n'
    file = open(out_path, 'a')
    file.write(row)
    file.close()


def early_stop(epoch, best_epoch, tol=17):
    if epoch - best_epoch > tol:
        return True
    else:
        return False






    
