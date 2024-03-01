import csv
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from torch import nn
from scipy.integrate import romb
from scipy.stats import beta




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
    samples[n] = torch.linspace(0.00001,0.99999,size).tolist()
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
        errors[i] -= np.std(samples[i]) * size *  train_size * std_w
        

    samples[n] = torch.linspace(0,1,size).tolist() # avoid propensity = 0 at t=0/1
    for i in errors:
        rank_error.append((errors[i], i))
    rank_error = sorted(rank_error)
    idx = rank_error[0][1]
    return torch.tensor(samples[idx]).unsqueeze(-1).float(), \
           torch.tensor(inv_density[idx]).reshape(1,size).float()
    
    

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
        x = batch['x'].float()
        t = torch.from_numpy(treatment_strengths).float()
        pre_y = model.get_predict(x,t)
        pred_dose_response = pre_y.flatten().detach().cpu().numpy()
        
        test_data = dict()
        patient = x[0].detach().cpu().numpy()
        test_data['x'] = np.repeat(np.expand_dims(patient, axis=0), num_integration_samples, axis=0)
        test_data['d'] = treatment_strengths # dosage

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


def export_result(out_path, Mise, num_unit, lr_main, lr_DA, hidden, num_grid1, \
                  t_grid, n_layer, dz, std_w):
    row = 'lr_main: ' + str(lr_main) + 'lr_DA: ' + str(lr_DA) + '_num_unit: ' + str(num_unit) + '_hidden: ' + \
          str(hidden) + '_num_grid: ' + str(num_grid1) + '_t_grid'+ str(t_grid)\
          + '_n_layer'+ str(n_layer) + '_dz'+ str(dz) + '_std_w' + str(std_w) + ' -- '
    row += 'MISE: (' + str(np.mean(Mise)) + ', ' + str(np.std(Mise)) + ')\n'
    file = open(out_path, 'a')
    file.write(row)
    file.close()


def early_stop(epoch, best_epoch, tol=17):
    if epoch - best_epoch > tol:
        return True
    else:
        return False






    
