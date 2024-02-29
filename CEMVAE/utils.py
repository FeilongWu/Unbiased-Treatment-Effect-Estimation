from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import torch
from sklearn.model_selection import train_test_split
import csv


class createDS(Dataset):
    def __init__(self, data, xdis, xcon):
        self.data = data
        self.xcon = xcon
        self.xcon_start = 5 + xdis
        self.xdis = xdis
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data[idx]
        dic = {'T':row[0]}
        dic['y_fac'] = row[1]
        dic['mu0'] = row[3]
        dic['mu1'] = row[4]
        dic['xdis'] = torch.tensor(row[5:5+self.xdis])
        dic['xcon'] = torch.tensor(row[self.xcon_start:self.xcon_start+self.xcon])
        return dic


class createDS_IHDP(Dataset):
    def __init__(self, data, xdis, xcon):
        self.data = data
        self.xcon = xcon
        self.xdis_start = 5 + xcon
        self.xdis = xdis
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data[idx]
        dic = {'T':row[0]}
        dic['y_fac'] = row[1]
        dic['mu0'] = row[3]
        dic['mu1'] = row[4]     
        dic['xdis'] = torch.tensor(row[self.xdis_start:self.xdis_start+self.xdis])
        dic['xcon'] = torch.tensor(row[5:5+self.xcon])
        return dic

class createDS_mixed(Dataset):
    def __init__(self, data, xdis, xcon):
        self.data = data
        self.xdis = xdis
        self.xcon = xcon
        self.xcon_start = 5 + xdis
        self.source = len(data)
    def __len__(self):
        return len(self.data[0])
    def __getitem__(self, idx):
        dic = {}
        for i in range(self.source):
            idx1 = str(i)
            row = self.data[i][idx]
            dic[idx1 + 'T'] = row[0]
            dic[idx1 + 'y_fac'] = row[1]
            dic[idx1 + 'mu0'] = row[3]
            dic[idx1 + 'mu1'] = row[4]
            dic[idx1 + 'xdis'] = torch.tensor(row[5:5+self.xdis])
            dic[idx1 + 'xcon'] = torch.tensor(row[self.xcon_start:self.xcon_start+self.xcon])
        return dic


class PreWhitener(nn.Module):
    """
    Data pre-whitener.
    """

    def __init__(self, data, device, store_mu_std=False):
        super().__init__()
        with torch.no_grad():
            loc = data.mean(0).to(device)
            scale = data.std(0).to(device)
            if store_mu_std:
                self.mean = loc.flatten().item()
                self.std = scale.flatten().item()
            scale[~(scale > 0)] = 1.0
            self.register_buffer("loc", loc)
            self.register_buffer("inv_scale", scale.reciprocal())

    def forward(self, data):
        return (data - self.loc) * self.inv_scale

    
def read_data(path):
    # data in csv
    data = []
    with open(path, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            temp = [] # treatment
            for i in row:
                temp.append(float(i))
            data.append(temp)
    return data

def read_data_IHDP(path):
    # data in csv
    data = []
    with open(path, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            temp = [] # treatment
            for i in row:
                temp.append(float(i))
            temp[18] -= 1
            data.append(temp)
    return data


@torch.no_grad()
def get_scores(model, data, device, prewhitener_x, prewhitener_y):
    true_ite, pre_ite = [], []
    for _, batch in enumerate(data):
        xcon = prewhitener_x(batch['xcon'].to(device))
        xdis = batch['xdis'].to(device)
        mu0 = batch['mu0'].to(device)
        mu1 = batch['mu1'].to(device)
        y0_pre, y1_pre = model.predict(xcon, xdis)
        y0_pre = y0_pre * prewhitener_y.std + prewhitener_y.mean
        y1_pre = y1_pre * prewhitener_y.std + prewhitener_y.mean
        true_ite.append(mu1 - mu0)
        pre_ite.append(y1_pre - y0_pre)
    true_ite = torch.cat(true_ite)
    pre_ite = torch.cat(pre_ite)
    ATE_err = abs(true_ite.mean(0) - pre_ite.mean(0))
    RMSE_PEHE = torch.sqrt(torch.mean((true_ite - pre_ite)**2))
    return float(ATE_err), float(RMSE_PEHE)

@torch.no_grad()   
def get_scores_qy(model, data, device, prewhitener_x, prewhitener_y):
    true_y, pre_y = [], []
    for _, batch in enumerate(data):
        xcon = prewhitener_x(batch['xcon'].to(device))
        xdis = batch['xdis'].to(device)
        mu0 = batch['mu0'].to(device)
        mu1 = batch['mu1'].to(device)
        #y0_pre, y1_pre = model.predict(xcon, xdis)
        x =  torch.cat((xcon, xdis), 1)
        y0_pre, _ = torch.chunk(model.qy0(x), 2, dim=1)
        y1_pre, _ = torch.chunk(model.qy1(x), 2, dim=1)
        y0_pre = y0_pre.flatten()
        y1_pre = y1_pre.flatten()
        y0_pre = y0_pre * prewhitener_y.std + prewhitener_y.mean
        y1_pre = y1_pre * prewhitener_y.std + prewhitener_y.mean
        true_y.append(mu0)
        true_y.append(mu1)
        pre_y.append(y0_pre)
        pre_y.append(y1_pre)
    true_y = torch.cat(true_y)
    pre_y = torch.cat(pre_y)
    RMSE = torch.sqrt(torch.mean((true_y - pre_y)**2))
    return float(RMSE)



@torch.no_grad()   
def get_scores_qy_within(model, data, device, prewhitener_x, prewhitener_y):
    true_y, pre_y = [], []
    for _, batch in enumerate(data):
        xcon = prewhitener_x(batch['xcon'].to(device))
        xdis = batch['xdis'].to(device)
        mu0 = batch['mu0'].to(device)
        mu1 = batch['mu1'].to(device)
        T = batch['T'].to(device)
        
        #y0_pre, y1_pre = model.predict(xcon, xdis)
        x =  torch.cat((xcon, xdis), 1)
        y0_pre, _ = torch.chunk(model.qy0(x), 2, dim=1)
        y1_pre, _ = torch.chunk(model.qy1(x), 2, dim=1)
        y0_pre = y0_pre.flatten()
        y1_pre = y1_pre.flatten()
        y0_pre = y0_pre * prewhitener_y.std + prewhitener_y.mean
        y1_pre = y1_pre * prewhitener_y.std + prewhitener_y.mean
        true_y.append(mu0*T + mu1 * (1-T))
        #true_y.append(mu1)
        pre_y.append(y0_pre*T + y1_pre*(1-T))
        #pre_y.append(y1_pre)
    true_y = torch.cat(true_y)
    pre_y = torch.cat(pre_y)
    RMSE = torch.sqrt(torch.mean((true_y - pre_y)**2))
    return float(RMSE)



@torch.no_grad()
def get_scores_target(model, data, device, prewhitener_x, prewhitener_y, target=''):
    true_ite, pre_ite = [], []
    for _, batch in enumerate(data):
        xcon = prewhitener_x(batch[str(target)+'xcon'].to(device))
        xdis = batch[str(target)+'xdis'].to(device)
        mu0 = batch[str(target)+'mu0'].to(device)
        mu1 = batch[str(target)+'mu1'].to(device)
        y0_pre, y1_pre = model.predict(xcon, xdis)
        y0_pre = y0_pre * prewhitener_y.std + prewhitener_y.mean
        y1_pre = y1_pre * prewhitener_y.std + prewhitener_y.mean
        true_ite.append(mu1 - mu0)
        pre_ite.append(y1_pre - y0_pre)
    true_ite = torch.cat(true_ite)
    pre_ite = torch.cat(pre_ite)
    ATE_err = abs(true_ite.mean(0) - pre_ite.mean(0))
    RMSE_PEHE = torch.sqrt(torch.mean((true_ite - pre_ite)**2))
    return float(ATE_err), float(RMSE_PEHE)

@torch.no_grad()
def get_scores_target_qy(model, data, device, prewhitener_x, prewhitener_y, target=''):
    true_y, pre_y = [], []
    for _, batch in enumerate(data):
        xcon = prewhitener_x(batch[str(target)+'xcon'].to(device))
        xdis = batch[str(target)+'xdis'].to(device)
        mu0 = batch[str(target)+'mu0'].to(device).unsqueeze(-1)
        mu1 = batch[str(target)+'mu1'].to(device).unsqueeze(-1)
        T = batch[str(target)+'T'].to(device).unsqueeze(-1)
        #y0_pre, y1_pre = model.predict(xcon, xdis)
        x  = torch.cat((xcon, xdis), 1)
        y0_pre, _ = torch.chunk(model.qy0(x), 2, dim=1)
        y1_pre, _ = torch.chunk(model.qy1(x), 2, dim=1)
        y0_pre = y0_pre * prewhitener_y.std + prewhitener_y.mean
        y1_pre = y1_pre * prewhitener_y.std + prewhitener_y.mean
        true_y.append((mu1 * T + mu0 * (1 - T)).flatten())
        pre_y.append((y1_pre * T + y0_pre * (1 - T)).flatten())
    true_y = torch.cat(true_y)
    pre_y = torch.cat(pre_y)
    RMSE = torch.sqrt(torch.mean((true_y - pre_y)**2))
    return float(RMSE)
        
