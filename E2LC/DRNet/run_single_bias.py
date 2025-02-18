import os
import sys
import numpy as np
import argparse
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from models.model_builder import ModelBuilder
from utils import *
#from apps.parameters import clip_percentage, parse_parameters
#from apps.evaluate import EvaluationApplication
#from apps.main import MainApplication
import csv
import json
import pickle



def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_treatments", default=1, type=int)
    parser.add_argument("--input_dim", default=10, type=int)
    parser.add_argument("--output_dim", default=1, type=int)
    parser.add_argument("--num_units", default=100, type=int)
    parser.add_argument("--dosage_samples", default=5, type=int)
    parser.add_argument("--num_exposure_strata", default=5, type=int)
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--learning_rate", default=0.0001, type=float)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--treatment_selection_bias", default=2.0, type=float)
    parser.add_argument("--dosage_selection_bias", default=2.0, type=float)
    parser.add_argument("--save_dataset", default=False)
    parser.add_argument("--validation_fraction", default=0.1, type=float)
    parser.add_argument("--test_fraction", default=0.2, type=float)
    parser.add_argument("--model_name", default="scigan_test")
    parser.add_argument("--verbose", default=False)
    parser.add_argument("--batch_size", default=150, type=int)
    parser.add_argument("--h_dim", default=64, type=int)
    parser.add_argument("--imbalance_loss_weight", default=1.0, type=int)
    parser.add_argument("--h_inv_eqv_dim", default=64, type=int)
    parser.add_argument("--num_dosage_samples", default=5, type=int)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--dataset", default='reg', type=str)

    return vars(parser.parse_args())



def load_data(dataset):

    x = []
    t = []
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
    t = np.array([0] * len(d))
    d = np.array(d)
    y = np.array(y)
    ids = np.array(ids)
    file = open('../data/' + dataset + '_propensity.pickle','rb')
    ps = pickle.load(file) # list
    return x, t, d, y, ids, ps


        
def scale_bias(ps, bias_level):
    ps = np.array(ps) ** bias_level
    ps = ps / sum(ps)
    return ps.tolist()



def data_split(x,t,d,y,ids, ps, test_ratio, num_treatments=1):
    n = len(d)
    ps_tr = []
    idx = np.arange(n)
    np.random.shuffle(idx)
    train_size = int(n * (1 - test_ratio))
    propensity = []
    for i in range(num_treatments):
        propensity.append(1 / num_treatments)
    data_tr = {'x':[], 't':[],'d':[],'y':[],'propensity':[],'ids':[]}
    data_te = {'x':[], 't':[],'d':[],'y':[],'ids':[]}
    for i in idx[:train_size]:
        data_tr['x'].append(x[i])
        data_tr['t'].append(t[i])
        data_tr['d'].append(d[i])
        data_tr['y'].append(y[i])
        data_tr['ids'].append(ids[i])
        data_tr['propensity'].append(propensity) # initialize propensities
        ps_tr.append(ps[i])

    for i in idx[train_size:]:
        data_te['x'].append(x[i])
        data_te['t'].append(t[i])
        data_te['d'].append(d[i])
        data_te['y'].append(y[i])
        data_te['ids'].append(ids[i])


    for i in data_tr:
        data_tr[i] = np.array(data_tr[i])
    for i in data_te:
        data_te[i] = np.array(data_te[i])

    return data_tr, data_te,ps_tr

class idx_generator:
    def __init__(self, size, bs,ps):
        self.idx = np.arange(0, size)
        self.bs = bs
        self.ps = (np.array(ps) / sum(ps)).tolist()
        

    def sample(self):
        return np.random.choice(self.idx, size=self.bs, p=self.ps)


    

def train(model, data_tr, args, train_steps, batch_idx_generator, best_model_path, \
          train_size, tolerance=12, bs=400):
    epochs = args['epochs']
    best_epoch = 0
    best_mse = np.inf
    for epoch in range(epochs):
        for step in range(train_steps):
            batch_idx = batch_idx_generator.sample()
            x, y = get_train_samples(data_tr, batch_idx)
            model.fit(x,y)
        propensity, pre_y = predict(model, data_tr, bs, train_size)
        data_tr['propensity'] = propensity
        mse = cal_mse(data_tr['y'], pre_y)
        if mse < best_mse:
            best_mse = mse
            best_epoch = epoch
            save_weights(model, best_model_path)
        if early_stop(epoch, best_epoch, tolerance):
            break
    weights = load_weights(best_model_path)
    model.set_weights(weights)
    return model

def export_result(out_path, Mise, DPE, PE, lr=0.001, num_unit=1, \
                num_layer=1, num_exposure_strata=1):
    row = 'lr: ' + str(lr) + '_num_unit: ' + str(num_unit) + '_num_layer: ' + \
          str(num_layer) + '_num_exposure_strata: ' + str(num_exposure_strata) + ' -- '
    row += 'MISE: (' + str(np.mean(Mise)) + ', ' + str(np.std(Mise)) + '), '
    row += 'DPE; (' + str(np.mean(DPE)) + ', ' + str(np.std(DPE)) + '), '
    row += 'PE; (' + str(np.mean(PE)) + ', ' + str(np.std(PE)) + ')\n'
    file = open(out_path, 'a')
    file.write(row)
    file.close()






if __name__ == '__main__':
    args = init_arg() # "seed" as a keyword
    seed = 909
    test_ratio = 0.2
    bias_level = 0.3 # base level = 0
    replications = 5
    dataset = 'mimiciv_coag'
    with open('../data/' + dataset + '_response_curve_calibrate.pickle', 'rb') as file:
        response_data = pickle.load(file)
    args = init_arg()
    if dataset in ['mimiciii_mv','mimiciii_mv10','mimiciii_mv20']:
        args['input_dim'] = 35
    elif dataset in ['mimiciv_seda','mimiciv_seda10','mimiciv_seda20']:
        args['input_dim'] = 25
    elif dataset in ['mimiciv_mv','mimiciv_mv10','mimiciv_mv20']:
        args['input_dim'] = 31
    elif dataset in ['mimic_vaso_m2','mimic_vaso_m6']:
        args['input_dim'] = 34
    elif dataset in ['USCMR_m2','USCMR_m6']:
        args['input_dim'] = 7
    elif dataset in ['mimiciii_seda','mimiciii_seda10','mimiciii_seda20']:
        args['input_dim'] = 34
    elif dataset in ['mimic_seda_m2','mimic_seda_m6']:
        args['input_dim'] = 12
    elif dataset in ['mimic_iv_micr_m2','mimic_iv_micr_m6']:
        args['input_dim'] = 23
    elif dataset in ['mimiciv_coag','mimiciv_coag10','mimiciv_coag20']:
        args['input_dim'] = 31
    elif dataset in ['mimic_diur_m2','mimic_diur_m6']:
        args['input_dim'] = 38
    elif dataset in ['eicu_mv_m2','eicu_mv_m6']:
        args['input_dim'] = 39
    args['imbalance_loss_weight'] = 0.0 # 1.0 = wasserstein, 0.0 = no
    if args['imbalance_loss_weight'] == 0.0:
        out_path = './DRNet_' + dataset + '_bias_level_' + str(int(bias_level*100)) + '.txt'
    else:
        out_path = './DRNet_' + dataset + '_bias_level_' + str(int(bias_level*100)) + '_Wasserstein.txt'
    file = open(out_path, 'w')
    file.write('')
    file.close()
    hyperparameters = {'mimiciii_mv':{'num_units':[40 ], 'lrs':[ 0.0001 ],\
                                'num_layers':[3], \
                           'num_exposure_stratas':[3 ]},\
                       'synthetic':{'num_units':[12,14,16], 'lrs':[0.001,0.0001],\
                                'num_layers':[2,3,4], \
                           'num_exposure_stratas':[3,4,5]},\
                       'mimiciv_mv':{'num_units':[34 ], 'lrs':[0.00005],\
                                'num_layers':[3], \
                           'num_exposure_stratas':[4]},\
                       'mimic_vaso_m6':{'num_units':[38, 58, 73], 'lrs':[0.001],\
                                'num_layers':[2,3,4], \
                           'num_exposure_stratas':[3,4,5]},\
                       'USCMR_m6':{'num_units':[10,15,20], 'lrs':[0.001],\
                                'num_layers':[2,3,4], \
                           'num_exposure_stratas':[3,4,5]},
                       'mimic_iv_seda_m6':{'num_units':[18,22,26], 'lrs':[0.001],\
                                'num_layers':[2,3,4], \
                           'num_exposure_stratas':[3,4,5]},\
                       'mimiciii_seda':{'num_units':[36], 'lrs':[0.0003],\
                                'num_layers':[3], \
                           'num_exposure_stratas':[3 ]},\
                       'mimiciv_seda':{'num_units':[ 36], 'lrs':[ 0.00005],\
                                'num_layers':[ 4], \
                           'num_exposure_stratas':[3 ]},\
                       'mimiciv_coag':{'num_units':[38 ], 'lrs':[ 0.0003 ],\
                                'num_layers':[ 4], \
                           'num_exposure_stratas':[4]},\
                       'mimic_diur_m6':{'num_units':[46,58,70], 'lrs':[0.001],\
                                'num_layers':[2,3,4], \
                           'num_exposure_stratas':[3,4,5]},\
                       'eicu_mv_m6':{'num_units':[52,62,72], 'lrs':[0.001],\
                                'num_layers':[2,3,4], \
                           'num_exposure_stratas':[3,4,5]}}[dataset]
    num_units = hyperparameters['num_units']
    lrs = hyperparameters['lrs']
    num_layers = hyperparameters['num_layers']
    num_exposure_stratas = hyperparameters['num_exposure_stratas']
    
    # hyperparameters; lr, hidden layers, hidden size, # strata
    x,t,d,y,ids,ps = load_data(dataset) 
    ps = scale_bias(ps, bias_level)
    for num_unit in num_units:
        for lr in lrs:
            for num_layer in num_layers:
                for num_exposure_strata in num_exposure_stratas:
                    np.random.seed(seed)
                    tf.random.set_random_seed(seed)
                    Mise, DPE, PE = [], [], []
                    for rep in range(replications):
                        
                        data_tr, data_te,ps_tr = data_split(x,t,d,y, ids,ps,test_ratio)
                        best_model_path = './checkpoints/' + dataset + '_DRNet.npz'
                        train_size = len(data_tr['y'])
                        train_steps = int(train_size / args['batch_size'])
                        batch_idx_generator = idx_generator(train_size, args['batch_size'],ps_tr)
                        args['num_units'] = num_unit
                        args['learning_rate'] = lr
                        args['num_layers'] = num_layer
                        args['num_exposure_strata'] = num_exposure_strata
                        args['dataset'] = dataset
    
    
                        model = ModelBuilder.build_tarnet(**args) # DRNet
                        best_model = train(model, data_tr, args, train_steps, batch_idx_generator, \
                        best_model_path, train_size)
                        mise, dpe, pe = evaluate_model(best_model, data_te,response_data, args['num_treatments'], \
                             args['dosage_samples'], dataset=dataset)
                        Mise.append(mise)
                        DPE.append(dpe)
                        PE.append(pe)
                    export_result(out_path, Mise, DPE, PE, lr=lr, num_unit=num_unit, \
                                 num_layer=num_layer, num_exposure_strata=num_exposure_strata)
                        
    



    
##import os
##import sys
##import numpy as np
##import argparse
##import tensorflow.compat.v1 as tf
##tf.disable_v2_behavior()
##from models.model_builder import ModelBuilder
##from utils import *
###from apps.parameters import clip_percentage, parse_parameters
###from apps.evaluate import EvaluationApplication
###from apps.main import MainApplication
##import csv
##import json
##import pickle
##
##
##
##def init_arg():
##    parser = argparse.ArgumentParser()
##    parser.add_argument("--num_treatments", default=1, type=int)
##    parser.add_argument("--input_dim", default=10, type=int)
##    parser.add_argument("--output_dim", default=1, type=int)
##    parser.add_argument("--num_units", default=100, type=int)
##    parser.add_argument("--dosage_samples", default=5, type=int)
##    parser.add_argument("--num_exposure_strata", default=5, type=int)
##    parser.add_argument("--num_layers", default=2, type=int)
##    parser.add_argument("--learning_rate", default=0.0001, type=float)
##    parser.add_argument("--epochs", default=500, type=int)
##    parser.add_argument("--treatment_selection_bias", default=2.0, type=float)
##    parser.add_argument("--dosage_selection_bias", default=2.0, type=float)
##    parser.add_argument("--save_dataset", default=False)
##    parser.add_argument("--validation_fraction", default=0.1, type=float)
##    parser.add_argument("--test_fraction", default=0.2, type=float)
##    parser.add_argument("--model_name", default="scigan_test")
##    parser.add_argument("--verbose", default=False)
##    parser.add_argument("--batch_size", default=150, type=int)
##    parser.add_argument("--h_dim", default=64, type=int)
##    parser.add_argument("--imbalance_loss_weight", default=1.0, type=int)
##    parser.add_argument("--h_inv_eqv_dim", default=64, type=int)
##    parser.add_argument("--num_dosage_samples", default=5, type=int)
##    parser.add_argument("--alpha", default=1.0, type=float)
##    parser.add_argument("--dataset", default='reg', type=str)
##
##    return vars(parser.parse_args())
##
##def load_data(dataset):
##
##    x = []
##    t = []
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
##    t = np.array([0] * len(d))
##    d = np.array(d)
##    y = np.array(y)
##    ids = np.array(ids)
##    file = open('../data/' + dataset + '_propensity.pickle','rb')
##    ps = pickle.load(file) # list
##    return x, t, d, y, ids, ps
##
##
##        
##def scale_bias(ps, bias_level):
##    ps = np.array(ps) ** bias_level
##    ps = ps / sum(ps)
##    return ps.tolist()
##
##
##
##def data_split(x,t,d,y,ids, ps, test_ratio, num_treatments=1):
##    n = len(d)
##    ps_tr = []
##    idx = np.arange(n)
##    np.random.shuffle(idx)
##    train_size = int(n * (1 - test_ratio))
##    propensity = []
##    for i in range(num_treatments):
##        propensity.append(1 / num_treatments)
##    data_tr = {'x':[], 't':[],'d':[],'y':[],'propensity':[],'ids':[]}
##    data_te = {'x':[], 't':[],'d':[],'y':[],'ids':[]}
##    for i in idx[:train_size]:
##        data_tr['x'].append(x[i])
##        data_tr['t'].append(t[i])
##        data_tr['d'].append(d[i])
##        data_tr['y'].append(y[i])
##        data_tr['ids'].append(ids[i])
##        data_tr['propensity'].append(propensity) # initialize propensities
##        ps_tr.append(ps[i])
##
##    for i in idx[train_size:]:
##        data_te['x'].append(x[i])
##        data_te['t'].append(t[i])
##        data_te['d'].append(d[i])
##        data_te['y'].append(y[i])
##        data_te['ids'].append(ids[i])
##
##
##    for i in data_tr:
##        data_tr[i] = np.array(data_tr[i])
##    for i in data_te:
##        data_te[i] = np.array(data_te[i])
##
##    return data_tr, data_te,ps_tr
##
##
##        
##    
##
##
##    
##
##def train(model, data_tr, args, train_steps, batch_idx_generator, best_model_path, \
##          train_size, tolerance=12, bs=400):
##    epochs = args['epochs']
##    best_epoch = 0
##    best_mse = np.inf
##    for epoch in range(epochs):
##        for step in range(train_steps):
##            batch_idx = batch_idx_generator.sample()
##            x, y = get_train_samples(data_tr, batch_idx)
##            model.fit(x,y)
##        propensity, pre_y = predict(model, data_tr, bs, train_size)
##        data_tr['propensity'] = propensity
##        mse = cal_mse(data_tr['y'], pre_y)
##        if mse < best_mse:
##            best_mse = mse
##            best_epoch = epoch
##            save_weights(model, best_model_path)
##        if early_stop(epoch, best_epoch, tolerance):
##            break
##    weights = load_weights(best_model_path)
##    model.set_weights(weights)
##    return model
##
##def export_result(out_path, Mise, DPE, PE, lr=0.001, num_unit=1, \
##                num_layer=1, num_exposure_strata=1):
##    row = 'lr: ' + str(lr) + '_num_unit: ' + str(num_unit) + '_num_layer: ' + \
##          str(num_layer) + '_num_exposure_strata: ' + str(num_exposure_strata) + ' -- '
##    row += 'MISE: (' + str(np.mean(Mise)) + ', ' + str(np.std(Mise)) + '), '
##    row += 'DPE; (' + str(np.mean(DPE)) + ', ' + str(np.std(DPE)) + '), '
##    row += 'PE; (' + str(np.mean(PE)) + ', ' + str(np.std(PE)) + ')\n'
##    file = open(out_path, 'a')
##    file.write(row)
##    file.close()
##
##
##class idx_generator:
##    def __init__(self, size, bs,ps):
##        self.idx = np.arange(0, size)
##        self.bs = bs
##        self.ps = (np.array(ps) / sum(ps)).tolist()
##        
##
##    def sample(self):
##        return np.random.choice(self.idx, size=self.bs, p=self.ps)
##    
##
##
##
##if __name__ == '__main__':
##    args = init_arg() # "seed" as a keyword
##    seed = 909
##    test_ratio = 0.2
##    replications = 5
##    bias_level = 0.1 # base level = 0
##    dataset = 'mimiciii_seda'
##    with open('../data/' + dataset + '_response_curve_calibrate.pickle', 'rb') as file:
##        response_data = pickle.load(file)
##    args = init_arg()
##    if dataset in ['mimiciii_mv','mimiciii_mv10','mimiciii_mv20']:
##        args['input_dim'] = 35
##    elif dataset in ['mimiciv_seda','mimiciv_seda10','mimiciv_seda20']:
##        args['input_dim'] = 25
##    elif dataset in ['mimiciv_mv','mimiciv_mv10','mimiciv_mv20']:
##        args['input_dim'] = 31
##    elif dataset in ['mimic_vaso_m2','mimic_vaso_m6']:
##        args['input_dim'] = 34
##    elif dataset in ['USCMR_m2','USCMR_m6']:
##        args['input_dim'] = 7
##    elif dataset in ['mimiciii_seda','mimiciii_seda10','mimiciii_seda20']:
##        args['input_dim'] = 34
##    elif dataset in ['mimic_seda_m2','mimic_seda_m6']:
##        args['input_dim'] = 12
##    elif dataset in ['mimic_iv_micr_m2','mimic_iv_micr_m6']:
##        args['input_dim'] = 23
##    elif dataset in ['mimiciv_coag','mimiciv_coag10','mimiciv_coag20']:
##        args['input_dim'] = 31
##    elif dataset in ['mimic_diur_m2','mimic_diur_m6']:
##        args['input_dim'] = 38
##    elif dataset in ['eicu_mv_m2','eicu_mv_m6']:
##        args['input_dim'] = 39
##    args['imbalance_loss_weight'] = 0.0 # 1.0 = wasserstein, 0.0 = no
##    if args['imbalance_loss_weight'] == 0.0:
##        out_path = './DRNet_' + dataset + '_bias_level_' + str(int(bias_level*100)) + '.txt'
##    else:
##        out_path = './DRNet_' + dataset + '_bias_level_' + str(int(bias_level*100)) + '_Wasserstein.txt'
##    file = open(out_path, 'w')
##    file.write('')
##    file.close()
##    #hyperparameters = {'mimiciii_mv':{'num_units':[40,44,48], 'lrs':[0.001,0.0003,0.0001,0.00005],\
##    #                            'num_layers':[2,3], \
##    #                       'num_exposure_stratas':[3,4,5]},\
##    #                   'synthetic':{'num_units':[12,14,16], 'lrs':[0.001,0.0001],\
##    #                            'num_layers':[2,3,4], \
##    #                       'num_exposure_stratas':[3,4,5]},\
##    #                   'mimiciv_mv':{'num_units':[34,38,42], 'lrs':[0.001,0.0003,0.0001,0.00005],\
##    #                            'num_layers':[2,3], \
##    #                       'num_exposure_stratas':[3,4,5]},\
##    #                   'mimic_vaso_m6':{'num_units':[38, 58, 73], 'lrs':[0.001],\
##    #                            'num_layers':[2,3,4], \
##    #                       'num_exposure_stratas':[3,4,5]},\
##    #                   'USCMR_m6':{'num_units':[10,15,20], 'lrs':[0.001],\
##    #                            'num_layers':[2,3,4], \
##    #                       'num_exposure_stratas':[3,4,5]},
##    #                   'mimic_iv_seda_m6':{'num_units':[18,22,26], 'lrs':[0.001],\
##    #                            'num_layers':[2,3,4], \
##    #                       'num_exposure_stratas':[3,4,5]},\
##    #                   'mimiciii_seda':{'num_units':[36,42,48], 'lrs':[0.001,0.0003,0.0001,0.00005],\
##    #                            'num_layers':[2,3,4], \
##    #                       'num_exposure_stratas':[3,4,5]},\
##    #                   'mimiciv_seda':{'num_units':[28,32,36], 'lrs':[0.001,0.0003,0.0001,0.00005],\
##    #                            'num_layers':[2,3,4], \
##    #                       'num_exposure_stratas':[3,4,5]},\
##    #                   'mimiciv_coag':{'num_units':[34,38,42], 'lrs':[0.001,0.0003,0.0001,0.00005],\
##    #                            'num_layers':[2,3,4], \
##    #                       'num_exposure_stratas':[3,4,5]},\
##    #                   'mimic_diur_m6':{'num_units':[46,58,70], 'lrs':[0.001],\
##    #                            'num_layers':[2,3,4], \
##    #                       'num_exposure_stratas':[3,4,5]},\
##    #                   'eicu_mv_m6':{'num_units':[52,62,72], 'lrs':[0.001],\
##    #                            'num_layers':[2,3,4], \
##    #                       'num_exposure_stratas':[3,4,5]}}[dataset]
##
##    hyperparameters = {'mimiciii_mv10':{'num_units':[40 ], 'lrs':[ 0.0001 ],\
##                                'num_layers':[3], \
##                           'num_exposure_stratas':[3 ]},\
##                       'synthetic':{'num_units':[12,14,16], 'lrs':[0.001,0.0001],\
##                                'num_layers':[2,3,4], \
##                           'num_exposure_stratas':[3,4,5]},\
##                       'mimiciv_mv20':{'num_units':[34 ], 'lrs':[0.00005],\
##                                'num_layers':[3], \
##                           'num_exposure_stratas':[4]},\
##                       'mimic_vaso_m6':{'num_units':[38, 58, 73], 'lrs':[0.001],\
##                                'num_layers':[2,3,4], \
##                           'num_exposure_stratas':[3,4,5]},\
##                       'USCMR_m6':{'num_units':[10,15,20], 'lrs':[0.001],\
##                                'num_layers':[2,3,4], \
##                           'num_exposure_stratas':[3,4,5]},
##                       'mimic_iv_seda_m6':{'num_units':[18,22,26], 'lrs':[0.001],\
##                                'num_layers':[2,3,4], \
##                           'num_exposure_stratas':[3,4,5]},\
##                       'mimiciii_seda':{'num_units':[36], 'lrs':[0.0003],\
##                                'num_layers':[3], \
##                           'num_exposure_stratas':[3 ]},\
##                       'mimiciv_seda20':{'num_units':[ 36], 'lrs':[ 0.00005],\
##                                'num_layers':[ 4], \
##                           'num_exposure_stratas':[3 ]},\
##                       'mimiciv_coag20':{'num_units':[38 ], 'lrs':[ 0.0003 ],\
##                                'num_layers':[ 4], \
##                           'num_exposure_stratas':[4]},\
##                       'mimic_diur_m6':{'num_units':[46,58,70], 'lrs':[0.001],\
##                                'num_layers':[2,3,4], \
##                           'num_exposure_stratas':[3,4,5]},\
##                       'eicu_mv_m6':{'num_units':[52,62,72], 'lrs':[0.001],\
##                                'num_layers':[2,3,4], \
##                           'num_exposure_stratas':[3,4,5]}}[dataset]
##    num_units = hyperparameters['num_units']
##    lrs = hyperparameters['lrs']
##    num_layers = hyperparameters['num_layers']
##    num_exposure_stratas = hyperparameters['num_exposure_stratas']
##    
##    # hyperparameters; lr, hidden layers, hidden size, # strata
##    x,t,d,y,ids,ps = load_data(dataset)
##    for num_unit in num_units:
##        for lr in lrs:
##            for num_layer in num_layers:
##                for num_exposure_strata in num_exposure_stratas:
##                    np.random.seed(seed)
##                    tf.random.set_random_seed(seed)
##                    Mise, DPE, PE = [], [], []
##                    for rep in range(replications):
##                        
##                        data_tr, data_te,ps_tr= data_split(x,t,d,y,ids, ps,test_ratio)
##                        best_model_path = './checkpoints/' + dataset + '_DRNet.npz'
##                        train_size = len(data_tr['y'])
##                        train_steps = int(train_size / args['batch_size'])
##                        batch_idx_generator = idx_generator(train_size, args['batch_size'],ps_tr)
##                        args['num_units'] = num_unit
##                        args['learning_rate'] = lr
##                        args['num_layers'] = num_layer
##                        args['num_exposure_strata'] = num_exposure_strata
##    
##    
##                        model = ModelBuilder.build_tarnet(**args) # DRNet
##                        best_model = train(model, data_tr, args, train_steps, batch_idx_generator, \
##                        best_model_path, train_size)
##                        mise, dpe, pe = evaluate_model(best_model, data_te, response_data, args['num_treatments'], \
##                             args['dosage_samples'],dataset)
##                        Mise.append(mise)
##                        DPE.append(dpe)
##                        PE.append(pe)
##                    export_result(out_path, Mise, DPE, PE, lr=lr, num_unit=num_unit, \
##                                 num_layer=num_layer, num_exposure_strata=num_exposure_strata)
##                        
##    
