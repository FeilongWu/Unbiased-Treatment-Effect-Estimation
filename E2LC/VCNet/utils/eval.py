import torch
import numpy as np
import json
from data.data import get_iter
from torch.utils.data import Dataset
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
        x1 = batch['x'].float()
        x = x1.repeat(num_integration_samples, 1)
        t = torch.from_numpy(treatment_strengths).float()
        _, pre_y = model.forward(t,x)
        pred_dose_response = pre_y.flatten().detach().cpu().numpy()
        
        test_data = dict()
        patient = x1[0].detach().cpu().numpy()
        test_data['x'] = np.repeat(np.expand_dims(patient, axis=0), num_integration_samples, axis=0)
        test_data['d'] = treatment_strengths # dosage

        true_outcomes = [get_patient_outcome(patient, v, d) for d in
                                 treatment_strengths]
        mise = romb(np.square(true_outcomes - pred_dose_response), dx=step_size)
        mises.append(mise)

    return np.sqrt(np.mean(mises))

##def curve(model, test_matrix, t_grid, targetreg=None):
##    n_test = t_grid.shape[1]
##    t_grid_hat = torch.zeros(2, n_test)
##    t_grid_hat[0, :] = t_grid[0, :]
##
##    test_loader = get_iter(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)
##
##    if targetreg is None:
##        for _ in range(n_test):
##            for idx, (inputs, y) in enumerate(test_loader):
##                t = inputs[:, 0]
##                t *= 0
##                t += t_grid[0, _]
##                x = inputs[:, 1:]
##                break
##            out = model.forward(t, x)
##            out = out[1].data.squeeze()
##            out = out.mean()
##            t_grid_hat[1, _] = out
##        mse = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2).mean().data
##        return t_grid_hat, mse
##    else:
##        for _ in range(n_test):
##            for idx, (inputs, y) in enumerate(test_loader):
##                t = inputs[:, 0]
##                t *= 0
##                t += t_grid[0, _]
##                x = inputs[:, 1:]
##                break
##            out = model.forward(t, x)
##            tr_out = targetreg(t).data
##            g = out[0].data.squeeze()
##            out = out[1].data.squeeze() + tr_out / (g + 1e-6)
##            out = out.mean()
##            t_grid_hat[1, _] = out
##        mse = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2).mean().data
##        return t_grid_hat, mse


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
