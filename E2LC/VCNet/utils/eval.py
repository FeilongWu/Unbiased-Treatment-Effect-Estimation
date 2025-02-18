import torch
import numpy as np
import json
from data.data import get_iter
from torch.utils.data import Dataset
from scipy.integrate import romb

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


##def MV(x,v,t):
##    mu = 4 * (t-0.5)**2*np.sin(np.pi/2*t) * 2 * \
##             ((sum(v[1]*x) / sum(v[2]*x))**0.5 + 10 * sum(v[0]*x))


##    return mu


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
    ##
    #pre_y1= np.zeros(65)
    #true_y = np.zeros(65)
    #count = 0
    ##
    for batch in data:
        x1 = batch['x'].float()
        idx = batch['ids'].float().item()
        
        x = x1.repeat(num_integration_samples, 1)
        t = torch.from_numpy(treatment_strengths).float()
        _, pre_y = model.forward(t,x)
        pred_dose_response = pre_y.flatten().detach().cpu().numpy()
        ##
        #pre_y1 += pred_dose_response
        #count  += 1  
        ##
        
        test_data = dict()
        patient = x1[0].detach().cpu().numpy()
        test_data['x'] = np.repeat(np.expand_dims(patient, axis=0), num_integration_samples, axis=0)
        test_data['d'] = treatment_strengths # dosage

        #true_outcomes = [target_function(patient, v, d) for d in
        #                        treatment_strengths]
        true_outcomes = response[idx]
        ##
        #true_y += np.array(true_outcomes)
        ##
        mise = romb(np.square(true_outcomes - pred_dose_response), dx=step_size)
        mises.append(mise)
    ##
    #print('predict',(pre_y1 / count).tolist())
    #print('true', (true_y/count).tolist())
    ##
    return np.sqrt(np.mean(mises))



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
