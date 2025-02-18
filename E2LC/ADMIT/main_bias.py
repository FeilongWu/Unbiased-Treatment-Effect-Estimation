from models.admit import *
from args import Helper
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils.train_helper import *
from utils.eval_helper import *
from utils.model_helper import *


from utils.data_helper import *
import os
import logging
import csv
import json
from sklearn.preprocessing import StandardScaler
import pickle
import csv




def load_data(dataset):
##    file = open('../data/' + dataset + '_metadata.json')
##    meta = json.load(file)
##    file.close()
##    v1 = meta['v1']
##    v2 = meta['v2']
##    v3 = meta['v3']

    
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
    file = open('../data/' + dataset + '_propensity.pickle','rb')
    ps = pickle.load(file) # list
    return x, d, y, ids, ps





def scale_bias(ps, bias_level):
    ps = np.array(ps) ** bias_level
    ps = ps / sum(ps)
    return ps.tolist()






def data_split(x,d,y,ids, ps, test_ratio, num_treatments=1):
    n = len(d)
    ps_tr = []
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
        ps_tr.append(ps[i])

    for i in idx[train_size:]:
        data_te['x'].append(x[i])
        data_te['d'].append(d[i])
        data_te['y'].append(y[i])
        data_te['ids'].append(ids[i])

    return data_tr, data_te, ps_tr







def export_result(out_path, Mise, num_unit, lr, k):
    row = 'lr: ' + str(lr) + '_num_unit: ' + str(num_unit) + '_k: ' + str(k) + ' -- '
    row += 'MISE: (' + str(np.mean(Mise)) + ', ' + str(np.std(Mise)) + ')\n'
    file = open(out_path, 'a')
    file.write(row)
    file.close()


def main(dataset='mimic'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    helper = Helper()
    args = helper.config
    args.device = device
    args.args_to_dict = helper.args_to_dict
    setup_seed(args.seed)

    logger = None
    if args.log:
        logger = logging.getLogger(__name__)
        logger.setLevel(level = logging.INFO)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        
        handler = logging.FileHandler("{}/log_{}.txt".format(args.log_dir, args.local_time))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.info(str(args.args_to_dict))


    test_ratio = 0.2
    batch_size = 150
    bias_level = 0.1 # base level = 0
    out_path = './ADMIT_' + dataset + '_bias_level_' + str(int(bias_level*100)) + '.txt'
    with open('../data/' + dataset + '_response_curve_calibrate.pickle', 'rb') as file:
        response_data = pickle.load(file)
    file = open(out_path, 'w')
    file.write('')
    file.close()
    hyperparameters = {'mimiciii_mv':{'num_units':[44], 'lrs':[0.00005],\
                                'k':[5]},\
                       'mimiciv_seda':{'num_units':[36], 'lrs':[0.00005],\
                                'k':[4]},\
                       'mimiciii_seda':{'num_units':[38], 'lrs':[0.0003],\
                                'k':[6]},\
                       'mimiciv_mv':{'num_units':[34], 'lrs':[0.001 ],\
                                'k':[5]},\
                       'mimiciv_coag':{'num_units':[38], 'lrs':[0.001],\
                                'k':[6]}}[dataset]
    replications = 5
    args.n_epochs = 500


    for num_unit in hyperparameters['num_units']:
        for lr in hyperparameters['lrs']:
            for k in hyperparameters['k']:
                args.learning_rate = lr
                Mise = []
                np.random.seed(3)
                for rep in range(replications):
                    torch.manual_seed(3)
                    x,t,y,ids,ps = load_data(dataset)

                    args.input_dim = x.shape[1]
                    if args.scale:
                        args.scaler = StandardScaler().fit(y.reshape(-1,1))
                    data_tr, data_te,ps_tr = data_split(x,t,y,ids,ps, test_ratio)
                    ps_tr = scale_bias(ps_tr, bias_level)
                    sampler = WeightedRandomSampler(ps_tr, batch_size, replacement=False if batch_size<len(ps_tr) else True)
                    data_tr = DataLoader(createDS(data_tr), batch_size=batch_size, sampler=sampler)
                    data_te = DataLoader(createDS(data_te), batch_size=1, shuffle=False)
    
    

                    model = ADMIT(args, h_unit=num_unit, dataset=dataset)
                    model.to(device)

                    model = train(model, data_tr, args, k=k)
                    mise = evaluate_model(model, data_te, response_data, args)
                    Mise.append(mise)
                export_result(out_path, Mise, num_unit, lr, k)
    

if __name__ == "__main__":
    dataset = 'mimiciii_seda'
    main(dataset=dataset)















##from models.admit import *
##from args import Helper
##from torch.utils.data import DataLoader, WeightedRandomSampler
##from utils.train_helper import *
##from utils.eval_helper import *
##from utils.model_helper import *
##import pickle
##
##
##from utils.data_helper import *
##import os
##import logging
##import csv
##import json
##from sklearn.preprocessing import StandardScaler
##import pickle
##
##
##
##
##def load_data(dataset):
####    file = open('../data/' + dataset + '_metadata.json')
####    meta = json.load(file)
####    file.close()
####    v1 = meta['v1']
####    v2 = meta['v2']
####    v3 = meta['v3']
##
##    
##    x = []
##    d = []
##    y = []
##    ids = []
##    file = '../data/' + dataset + '.csv'
##    with open(file) as file1:
##        reader = csv.reader(file1, delimiter=',')
##        for row in reader:
##            #t.append(int(row[0]))
##            d.append(float(row[1]))
##            y.append(float(row[0]))
##            ids.append(float(row[-1]))
##            temp = []
##            for entry in row[2:-1]:
##                temp.append(float(entry))
##            x.append(temp)
##    x = np.array(x)
##    d = np.array(d)
##    y = np.array(y)
##    ids = np.array(ids)
##    file = open('../data/' + dataset + '_propensity.pickle','rb')
##    ps = pickle.load(file) # list
##    return x, d, y, ids, ps
##
##
##def scale_bias(ps, bias_level):
##    ps = np.array(ps) ** bias_level
##    ps = ps / sum(ps)
##    return ps.tolist()
##
##
##def data_split(x,d,y,ids, ps, test_ratio, num_treatments=1):
##    n = len(d)
##    ps_tr = []
##    idx = np.arange(n)
##    np.random.shuffle(idx)
##    train_size = int(n * (1 - test_ratio))
##    propensity = []
##    data_tr = {'x':[], 't':[],'d':[],'y':[], 'ids':[]}
##    data_te = {'x':[], 't':[],'d':[],'y':[], 'ids':[]}
##    for i in idx[:train_size]:
##        data_tr['x'].append(x[i])
##        data_tr['d'].append(d[i])
##        data_tr['y'].append(y[i])
##        data_tr['ids'].append(ids[i])
##        ps_tr.append(ps[i])
##
##    for i in idx[train_size:]:
##        data_te['x'].append(x[i])
##        data_te['d'].append(d[i])
##        data_te['y'].append(y[i])
##        data_te['ids'].append(ids[i])
##
##    return data_tr, data_te, ps_tr
##
##
##def export_result(out_path, Mise, num_unit, lr, k):
##    row = 'lr: ' + str(lr) + '_num_unit: ' + str(num_unit) + '_k: ' + str(k) + ' -- '
##    row += 'MISE: (' + str(np.mean(Mise)) + ', ' + str(np.std(Mise)) + ')\n'
##    file = open(out_path, 'a')
##    file.write(row)
##    file.close()
##
##
##def main(dataset='mimic'):
##    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
##    helper = Helper()
##    args = helper.config
##    args.device = device
##    args.args_to_dict = helper.args_to_dict
##    setup_seed(args.seed)
##
##    logger = None
##    if args.log:
##        logger = logging.getLogger(__name__)
##        logger.setLevel(level = logging.INFO)
##        if not os.path.exists(args.log_dir):
##            os.makedirs(args.log_dir)
##        
##        handler = logging.FileHandler("{}/log_{}.txt".format(args.log_dir, args.local_time))
##        handler.setLevel(logging.INFO)
##        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
##        handler.setFormatter(formatter)
##        logger.addHandler(handler)
##        logger.info(str(args.args_to_dict))
##
##
##    test_ratio = 0.2
##    batch_size = 150
##    bias_level = 0.1 # base level = 0
##    out_path = './ADMIT_' + dataset + '_bias_level_' + str(int(bias_level*100)) + '.txt'
##    with open('../data/' + dataset + '_response_curve_calibrate.pickle', 'rb') as file:
##        response_data = pickle.load(file)
##    file = open(out_path, 'w')
##    file.write('')
##    file.close()
##    hyperparameters = {'mimic':{'num_units':[40,50,60], 'lrs':[0.001, 0.0001, 0.0002],\
##                                'k':[4,5,6]},\
##                       'eicu':{'num_units':[40,50,60], 'lrs':[0.001, 0.0001, 0.0002],\
##                                'k':[4,5,6]},\
##                       'ihdp':{'num_units':[35,45,55], 'lrs':[0.001, 0.0001, 0.0002],\
##                                'k':[4,5,6]}}[dataset]
##    replications = 5
##    args.n_epochs = 1000
##
##
##    for num_unit in hyperparameters['num_units']:
##        for lr in hyperparameters['lrs']:
##            for k in hyperparameters['k']:
##                args.learning_rate = lr
##                Mise = []
##                np.random.seed(3)
##                for rep in range(replications):
##                    torch.manual_seed(3)
##                    x,t,y,ids,ps = load_data(dataset)
##                    ps = scale_bias(ps, bias_level)
##                    args.input_dim = x.shape[1]
##                    if args.scale:
##                        args.scaler = StandardScaler().fit(y.reshape(-1,1))
##                    data_tr, data_te,ps_tr = data_split(x,t,y,ids,ps, test_ratio)
##                    sampler = WeightedRandomSampler(ps_tr, batch_size, replacement=False if batch_size<len(ps_tr) else True)
##                    data_tr = DataLoader(createDS(data_tr), batch_size=batch_size, sampler=sampler)
##                    data_te = DataLoader(createDS(data_te), batch_size=1, shuffle=False)
##    
##    
##
##                    model = ADMIT(args, h_unit=num_unit, dataset=dataset)
##                    model.to(device)
##
##                    model = train(model, data_tr, args, k=k)
##                    mise = evaluate_model(model, data_te, response_data, args)
##                    Mise.append(mise)
##                export_result(out_path, Mise, num_unit, lr, k)
##    
##
##if __name__ == "__main__":
##    dataset = 'mimic'
##    main(dataset=dataset)
