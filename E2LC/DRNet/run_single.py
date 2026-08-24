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


def load_data(dataset,rep, gamma=None):    
    x = []
    d = []
    t = []
    y = []
    ids = []
    response = {}
    if gamma is None:
        file = '../data/' + dataset + '_rep' + str(rep) + '.csv'
    else:
        file = '../data/' + dataset + '_rep' + str(rep) + '_gamma' + str(gamma) + '.csv'
        
    with open(file) as file1:
        reader = csv.reader(file1, delimiter=',')
        count = 0
        for row in reader:
            #t.append(int(row[0]))
            d.append(float(row[1]))
            y.append(float(row[0]))
            ids.append(count)
            temp = []
            for entry in row[2:]:
                temp.append(float(entry))
            x.append(temp)
            count += 1
    x = np.array(x)
    t = np.array([0] * len(d))
    d = np.array(d)
    y = np.array(y)
    if gamma is None:
        file = '../data/' + dataset + '_response_rep' + str(rep) + '.csv'
    else:
        file = '../data/' + dataset + '_response_rep' + str(rep) + '_gamma' + str(gamma) + '.csv'
    
    with open(file) as file1:
        reader = csv.reader(file1, delimiter=',')
        count = 0
        for row in reader:
            temp = []
            for entry in row:
                temp.append(float(entry))
            response[count] = np.array(temp)
            count += 1
    return x, t, d, y, ids, response


        
    


    

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
    replications = 2
    dataset = 'synthetic'
    gamma = None
    args = init_arg()
    if 'synthetic' in dataset :
        args['input_dim'] = 6
    elif 'news' in dataset:
        args['input_dim'] = 3477
    elif 'mimiciii' in dataset:
        args['input_dim'] = 31
    
    args['imbalance_loss_weight'] = 0.0 # 1.0 = wasserstein, 0.0 = no
    if args['imbalance_loss_weight'] == 0.0:
        out_path = './DRNet_' + dataset + '.txt'
    else:
        out_path = './DRNet_' + dataset + '_Wasserstein.txt'
    file = open(out_path, 'w')
    file.write('')
    file.close()


    hyperparameters = {'synthetic':{'num_units':[10,14,18 ], 'lrs':[0.001,0.0001,0.00001],\
                                'num_layers':[2,3],'num_exposure_stratas':[3,4,5]},
                       'news':{'num_units':[430,530,630 ], 'lrs':[0.001,0.0001,0.00001],\
                                'num_layers':[2,3],'num_exposure_stratas':[3,4,5]},
                       'mimiciii':{'num_units':[27,37,47], 'lrs':[0.001,0.0001,0.00001],\
                                'num_layers':[2,3],'num_exposure_stratas':[3,4,5]},
                       'tcga':{'num_units':[460,560,660], 'lrs':[0.001,0.0001,0.00001],\
                                'num_layers':[2,3],'num_exposure_stratas':[3,4,5]}}[dataset]
    num_units = hyperparameters['num_units']
    lrs = hyperparameters['lrs']
    num_layers = hyperparameters['num_layers']
    num_exposure_stratas = hyperparameters['num_exposure_stratas']
    
    # hyperparameters; lr, hidden layers, hidden size, # strata
    
    for num_unit in num_units:
        for lr in lrs:
            for num_layer in num_layers:
                for num_exposure_strata in num_exposure_stratas:
                    np.random.seed(seed)
                    tf.random.set_random_seed(seed)
                    Mise, DPE, PE = [], [], []
                    for rep in range(replications):
                        x,t,d,y,ids,response_data = load_data(dataset, rep, gamma=gamma)
                        data_tr, data_te = data_split(x,t,d,y,ids, test_ratio)
                        best_model_path = './checkpoints/' + dataset + '_DRNet.npz'
                        train_size = len(data_tr['y'])
                        train_steps = int(train_size / args['batch_size'])
                        batch_idx_generator = idx_generator(train_size, args['batch_size'])
                        args['num_units'] = num_unit
                        args['learning_rate'] = lr
                        args['num_layers'] = num_layer
                        args['num_exposure_strata'] = num_exposure_strata
    
    
                        model = ModelBuilder.build_tarnet(**args) # DRNet
                        best_model = train(model, data_tr, args, train_steps, batch_idx_generator, \
                        best_model_path, train_size)
                        mise, dpe, pe = evaluate_model(best_model, data_te, response_data, args['num_treatments'], \
                             args['dosage_samples'],dataset)
                        Mise.append(mise)
                        DPE.append(dpe)
                        PE.append(pe)
                    export_result(out_path, Mise, DPE, PE, lr=lr, num_unit=num_unit, \
                                 num_layer=num_layer, num_exposure_strata=num_exposure_strata)
                        
    


    '''
import numpy as np
from models.model_builder import ModelBuilder
nn = ModelBuilder.build_tarnet(2,1)
x = np.array([[1.2,3.5]])
t = np.array([2])
d = np.array([0.7])
y = np.array([6])
nn.predict([x,t,d])
propensity = np.array([[0.3,0.5,0.2]])
nn.fit([x,t,d], [propensity,y])
    '''
    
