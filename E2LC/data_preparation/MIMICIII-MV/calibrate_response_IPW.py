import numpy as np
import csv
import matplotlib.pyplot as plt
import pickle
import scipy.stats
import pickle
import torch
from torch import nn
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
from pyro.nn import DenseNN, ConditionalAutoRegressiveNN
from torch.utils.data import Dataset, DataLoader
import copy 
import os
from scipy.optimize import curve_fit
import scipy.stats


'''
The implementation use the read data to determine the counterfactuals
by selecting top k% of patients.
add penalty function.
'''

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def CNF_best_loss(model, data_tr, path, lr, tol=12):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = np.inf
    for epoch in range(300):
        cum_loss = 0
        for idx, batch in enumerate(data_tr):
            x=batch['x'].float()
            t=batch['t'].unsqueeze(-1).float()
            t -= 0.5
            t += torch.randn_like(t) * 0.1
            t = torch.clip(t, -0.5, 0.5)
            loss = -model.log_prob(t, x).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            cum_loss += loss.item()
        if cum_loss < best_loss:
           best_loss = cum_loss
           best_epoch = epoch
           torch.save(model.state_dict(), path)
        if early_stop(epoch, best_epoch, tol=tol):
           break
    return float(best_loss)

def early_stop(epoch, best_epoch, tol=17):
    if epoch - best_epoch > tol:
        return True
    else:
        return False

    
def train_CNF(model, data_tr, path, tol=12, lrs = [0.001,0.0003,0.0001,0.00005]):
    # choose best lr
    lr_loss = {}
    for lr in lrs:
        lr_loss[lr] = CNF_best_loss(copy.deepcopy(model), data_tr, path, lr)
    loss_lr = []
    for i in lr_loss:
        loss_lr.append((lr_loss[i],i))
    best_lr = sorted(loss_lr)[0][1]
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
    best_loss = np.inf
    for epoch in range(300):
        cum_loss = 0
        for idx, batch in enumerate(data_tr):
            x=batch['x'].float()
            t=batch['t'].unsqueeze(-1).float()
            t -= 0.5
            t += torch.randn_like(t) * 0.1
            t = torch.clip(t, -0.5, 0.5)
            loss = -model.log_prob(t, x).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            cum_loss += loss.item()
        if cum_loss < best_loss:
           best_loss = cum_loss
           best_epoch = epoch
           torch.save(model.state_dict(), path)
        if early_stop(epoch, best_epoch, tol=tol):
           break
    model.load_state_dict(torch.load(path))
    return model


class ConditionalNormalizingFlow(nn.Module):
    def __init__(self, input_dim, split_dim, context_dim, hidden_dim, num_layers, flow_length, 
                count_bins, order, bound, use_cuda):
        super(ConditionalNormalizingFlow, self).__init__()
        self.input_dim = input_dim
        self.split_dim = split_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.flow_length = flow_length
        self.count_bins = count_bins
        self.order = order
        self.bound = bound

        self.device = 'cpu' if not use_cuda else 'cuda'
        
        self.has_prop_score = False
        self.cond_base_dist = dist.MultivariateNormal(torch.zeros(self.input_dim).float(),
                                                      torch.diag(torch.ones(self.input_dim)).float())

        self.cond_loc = torch.nn.Parameter(torch.zeros((self.input_dim, )).float())
        self.cond_scale = torch.nn.Parameter(torch.ones((self.input_dim, )).float())
        self.cond_affine_transform = T.AffineTransform(self.cond_loc, self.cond_scale)


        if self.input_dim == 1:
            self.cond_spline_nn = DenseNN(
                                        self.context_dim,
                                         [self.hidden_dim],
                                          param_dims=[self.count_bins, 
                                                        self.count_bins,
                                                      (self.count_bins - 1)]).float()
            self.cond_spline_transform = [T.ConditionalSpline(self.cond_spline_nn,
                                                            self.input_dim,
                                                             order='quadratic',
                                                             count_bins=self.count_bins,
                                                             bound=self.bound).to(self.device) for _ in range(self.flow_length)]
        else:
            self.cond_spline_nn = ConditionalAutoRegressiveNN(self.input_dim,
                                                               self.context_dim, 
                                                              [self.hidden_dim],
                                                              param_dims=[self.count_bins,
                                                                          self.count_bins,
                                                                          (self.count_bins - 1)]).float()
            self.cond_spline_transform = [T.ConditionalSplineAutoregressive(self.input_dim,
                                                                           self.cond_spline_nn,
                                                                           order='quadratic',
                                                                           count_bins=self.count_bins,
                                                                           bound=self.bound).to(self.device) for _ in range(self.flow_length)]
        self.flow_dist = dist.ConditionalTransformedDistribution(self.cond_base_dist,
                                                                      [self.cond_affine_transform] + self.cond_spline_transform) #[self.cond_affine_transform, self.cond_spline_transform]

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()
            nn.ModuleList(self.transforms).cuda()
            self.base_dist = dist.Normal(torch.zeros(input_dim).cuda(),
                                         torch.ones(input_dim).cuda())
    
    def sample(self, H, num_samples=1):
        assert num_samples >= 1
        num_H = H.shape[0] if len(H.shape)==2 else 1
        dim_samples = [num_samples, num_H] if (num_samples > 1 and num_H > 1) else [num_H] if num_H > 1 else [num_samples]
        x = self.flow_dist.condition(H).sample(dim_samples)
        return x
    
    def log_prob(self, x, H):
        # x = x.reshape(-1, self.input_dim)
        cond_flow_dist = self.flow_dist.condition(H) 
        # print(x.shape, H.shape)
        return cond_flow_dist.log_prob(x)

    def model(self, X=None, H=None):
        N = len(X) if X is not None else None
        pyro.module("nf", nn.ModuleList(self.cond_spline_transform))
        with pyro.plate("data", N):
                self.cond_flow_dist = self.flow_dist.condition(H)
                obs = pyro.sample("obs", self.cond_flow_dist, obs=X)
            
    def guide(self, X=None, H=None):
        pass


    
def get_weights(data):
    y = data[:,0]
    w = []
    col = data.shape[1] - 1
    for i in range(2,col):
        w.append(scipy.stats.pearsonr(data[:,i], y)[0])
    return np.array(w)

def get_similarity(x1,x2,w):
    dist = (x1 - x2) ** 2
    sim = 1 / (dist + 0.001)
    return sum(sim * w) ** 0.5

def find_top_k(data, x, feature_w):
    t,y = [], []
    for regime in data:
        sim_t_y = []
        for i in data[regime]:
            sim = get_similarity(x, i[2:-1], feature_w)
            sim_t_y.append((sim, (i[1], i[0])))
        temp_t, temp_y, temp_sim = [], [], []
        for sim, ty in sim_t_y:
            temp_sim.append(sim)
            temp_t.append(ty[0])
            temp_y.append(ty[1])
        total_sim = np.sum(temp_sim)
        t.append(np.sum(np.array(temp_t) * np.array(temp_sim)) / total_sim)
        y.append(np.sum(np.array(temp_y) * np.array(temp_sim)) / total_sim)
    return np.array(t), np.array(y)


def data_to_regime(data1, regimes):
    if type(regimes) is int:
        steps = np.linspace(0,1,regimes+1)
        regimes = []
        for i in range(0,len(steps)-1):
            if i == len(steps)-2:
                regimes.append((steps[i], steps[i+1]+0.00001))
            else:
                regimes.append((steps[i], steps[i+1]))
            
    data = {} # data by discretized regimes
    for i in regimes:
        data[i] = []
    for i in data1:
        t = i[1]
        for key in data:
            if key[0] <= t and t < key[1]:
                data[key].append(i)
                break
    return data



def fit_curve1(fun, penalty_fun,initial_parameters, t, y,target_t,\
               upper=10,lower=0):
    not_find_params = True
    while not_find_params:
        try:
            fittedParameters, _ = curve_fit(penalty_fun, t, y, \
                                        initial_parameters,maxfev = 6000)
            y1 =  np.clip(fun(target_t, *fittedParameters), lower,upper)
            not_find_params = False
        except:
            initial_parameters = np.random.normal(0,0.1,4)
    return y1

def cal_penalty(x, con, scale=1000):
    if (con * x) < 0:
        return abs(x) * scale
    else:
        return 0
    


def fun(t, a, b, c, d):
    return a * t ** 3 + b * t ** 2 + c * t + d
    #return a * t ** 2 + b * t + d * t ** 0.5 + c

def penalty_fun(x,a,b,c,d):
    constrain = [0,0,0,0] # [0,0,0,0] for softplus, [0,1,0,1] for polynominals
    # need to manually change penalty_fun.
    # polynominal is better than softplus.
    inputs = [x,a,b,c,d]
    output = fun(*inputs)
    penalty = 0
    for idx, con in enumerate(constrain):
        penalty += cal_penalty(inputs[idx+1], con)
    return output + penalty

def standardize(x):
    x = np.array(x)
    return (x - np.mean(x)) / np.std(x) / 10
    
def fit_curve(data, t, density_model,upper=10,lower=0,\
              ty_min_max=[0.0,1.0,0.0,1.0], plot_single=False, use_IPW=False,\
              patient_idx=0, regimes=3):
    constrain = [0,0,0,0]
    data1 = np.array(data)
    feature_w = get_weights(data1)
    data2 = data_to_regime(data1, regimes)
    result = {}
    #coef = 1.0
    #initial_parameters = np.array([coef, 0.1, 0.1, 0.1])
    t_min, t_max_min,y_mu,y_std = ty_min_max
    if plot_single:
        x = data1[patient_idx][2:-1]
        top_k_t, top_k_y = find_top_k(data2, x, feature_w**2)
        plt.scatter(np.array(top_k_t) * t_max_min + t_min, \
                    np.array(top_k_y) * y_std + y_mu)

        initial_parameters = np.array([0.1, 0.1, 0.1, 0.0])
        ###
        if use_IPW:
            ### top_k_y reweight
            x = torch.tensor(x).float().repeat(len(top_k_t), 1)
            t1 = np.clip(torch.tensor(top_k_t).float(),0.017,1.0).\
                     reshape(len(top_k_t),1) - 0.5
            temp = (np.exp(density_model.log_prob(t1, x).flatten()\
                               .detach().numpy())) ** 0.5
            temp = np.clip(temp / np.mean(temp), 0.005, 5)
            top_k_y = np.array(top_k_y) / temp

            y1 = fit_curve1(fun, penalty_fun, initial_parameters, top_k_t, top_k_y,\
                                t,upper=upper,lower=lower)
        else:# no IPW
            y1 = fit_curve1(fun, penalty_fun,initial_parameters, top_k_t, top_k_y,\
                                t,upper=upper,lower=lower)
        result[patient_idx] = np.array(y1)
    else:
        if use_IPW:
            save_pth = 'mimiciii_mv_response_curve_calibrate_IPW.pickle'
            if os.path.isfile(save_pth):
                with open(save_pth, 'rb') as file:
                    result = pickle.load(file)
                    start = len(result) + 1
            else:
                start = 0
        else:
            save_pth = 'mimiciii_mv_response_curve_calibrate.pickle'
            if os.path.isfile(save_pth):
                with open(save_pth, 'rb') as file:
                    result = pickle.load(file)
                    start = len(result) + 1
            else:
                start = 0
        
        keys = sorted(data1[:,-1])
        for idx in range(start, len(keys)):
            x = data1[idx][2:-1]
            top_k_t, top_k_y = find_top_k(data2, x, feature_w**2) # sorted
            

            initial_parameters = np.array([0.1, 0.1, 0.1, 0.0])
            ###
            
            if use_IPW:
            ### top_k_y reweight
                x = torch.tensor(x).float().repeat(len(top_k_t), 1)
                t1 = np.clip(torch.tensor(top_k_t).float(),0.017,1.0).\
                     reshape(len(top_k_t),1) - 0.5
                temp = (np.exp(density_model.log_prob(t1, x).flatten()\
                               .detach().numpy())) ** 0.5
                temp = np.clip(temp / np.mean(temp), 0.005, 5)
                top_k_y = np.array(top_k_y) / temp
                y1 = fit_curve1(fun, penalty_fun,initial_parameters, top_k_t, top_k_y,\
                                t,upper=upper,lower=lower)
                result[idx] = np.array(y1)
        
            else: # no IPW
                y1 = fit_curve1(fun, penalty_fun,initial_parameters, top_k_t, top_k_y,\
                                t,upper=upper,lower=lower)
                result[idx] = np.array(y1)
            if idx % 100 == 0 or idx == len(keys)-1:
                with open(save_pth, 'wb') as file:
                    pickle.dump(result, file)
    return result




def extract_max_min_t(path, threshold=0.03):
    #threshold: outlier threshold
    data = {}
    include_col = [0, 15, 10, 7] #[subject_id, y, t, feature]
    with open(path) as file:
        reader = csv.reader(file, delimiter=',')
        row_idx = 0
        for row in reader:
            if row_idx == 0:
                row_idx = 1
                continue
            stay_id = int(float(row[2]))
            data[stay_id] = []
            for i in include_col:
                data[stay_id].append(float(row[i]))
    # exclude outlier dosage and label
    dosage = []
    outcome = []
    for i in data:
        dosage.append(data[i][2])
        outcome.append(data[i][1])
    dosage = sorted(dosage)
    outcome = sorted(outcome)
    dosage_out = dosage[int(len(dosage)*(1-threshold))]
    y_out = outcome[int(len(outcome)*(1-threshold))]
    
    data1 = {}
    for i in data:
        if data[i][2] < dosage_out and data[i][1] < y_out:
            data1[i] = data[i]
    t = []
    for i in data1:
        t.append(data1[i][2])
    return min(t), max(t) - min(t)


def load_data(path):
    data = []
    ids = []
    with open(path) as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            temp = []
            for entry in row[:-1]:
                temp.append(float(entry))
            data.append(temp)
            ids.append([float(row[-1])])
    data = np.array(data)
    ids = np.array(ids)
    return np.concatenate((data, ids), axis=1)

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

    
if __name__ == '__main__':
    path_lable = './mimiciii_mv.csv'
    t_min, t_max_min = 4.639175257, 5.924205
    y_mu = 258.5053711
    y_std = 125.003714487

    path = './mimiciii_mv.csv'
    data = load_data(path) # [y,t,features, ids]
    dx = len(data[0]) - 3
    samples_power_of_two = 6
    num_integration_samples = 2 ** samples_power_of_two + 1
    t = np.linspace(np.finfo(float).eps, 1, num_integration_samples)

    density_model = ConditionalNormalizingFlow(input_dim=1, split_dim=0, \
                                            context_dim=dx, hidden_dim=int(1.2*dx), \
                                            num_layers=2, flow_length=1, \
                                            count_bins=5, order='quadratic', \
                                            bound=0.5, use_cuda=False)

    data_tr1 = {'y':data[:,0], 'd':data[:,1],'x':data[:,2:-1], 'ids':data[:,-1]}
    data_tr  = DataLoader(createDS(data_tr1), batch_size=150, shuffle=True)
    data_te  = DataLoader(createDS(data_tr1), batch_size=1, shuffle=False)
    density_model_path = './CNF.pth'
    torch.manual_seed(3)
    if os.path.isfile(density_model_path):
        density_model.load_state_dict(torch.load(density_model_path))
        print('Load density model from checkpoint')
    else:
        density_model = train_CNF(density_model, data_tr, \
                            density_model_path)



    plot_single = True
    patient_idx = 0 # this is used only when plot_single = True
                    # otherwise, plot average all
    use_IPW = False

    regimes = [(0.0,0.2),(0.2,0.5),(0.5,0.7),(0.7,0.80),(0.80,0.85), (0.95,1.0)]
    


    
    response_curve = fit_curve(data, t, density_model, \
                               lower=-5.0, upper=5.0, \
                               ty_min_max=[t_min, t_max_min,y_mu,y_std],\
                               plot_single=plot_single,use_IPW=use_IPW,\
                               patient_idx=patient_idx,regimes=regimes)



    
