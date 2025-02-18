import argparse
import os
import shutil
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np

from data_simulation import get_dataset_splits, TCGA_Data,get_split_indices
from SCIGAN_bias import SCIGAN_Model
from utils.evaluation_utils import compute_eval_metrics
import pickle
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_treatments", default=1, type=int)
    parser.add_argument("--treatment_selection_bias", default=2.0, type=float)
    parser.add_argument("--dosage_selection_bias", default=2.0, type=float)
    parser.add_argument("--save_dataset", default=False)
    parser.add_argument("--validation_fraction", default=0.1, type=float)
    parser.add_argument("--test_fraction", default=0.2, type=float)
    parser.add_argument("--model_name", default="scigan_test")
    parser.add_argument("--verbose", default=False)
    parser.add_argument("--batch_size", default=150, type=int)
    parser.add_argument("--h_dim", default=60, type=int)
    parser.add_argument("--h_inv_eqv_dim", default=60, type=int)
    parser.add_argument("--num_dosage_samples", default=6, type=int)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--epochs", default=5, type=int)

    return parser.parse_args()


def load_data(dataset):
    with open('../data/' + dataset + '_response_curve_calibrate.pickle', 'rb') as file:
        response_data = pickle.load(file)


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
    ps = np.array(pickle.load(file)) # list
    return x, t, d, y, ids, ps, response_data


def scale_bias(ps, bias_level):
    ps = np.array(ps) ** bias_level
    ps = ps / sum(ps)
    return ps


class TCGA_Data():
    def __init__(self, args, datasetname):
        np.random.seed(3)

        self.num_treatments = args['num_treatments']
        self.treatment_selection_bias = args['treatment_selection_bias']
        self.dosage_selection_bias = args['dosage_selection_bias']

        self.validation_fraction = args['validation_fraction']
        self.test_fraction = args['test_fraction']

        #self.tcga_data = pickle.load(open('datasets/tcga.p', 'rb'))
        #self.patients = self.normalize_data(self.tcga_data['rnaseq'])

        self.scaling_parameteter = 10
        self.datasetname = datasetname
        self.noise_std = 0.2

        self.num_weights = 3
##        self.v = np.zeros(shape=(self.num_treatments, self.num_weights, self.patients.shape[1]))
##
##        for i in range(self.num_treatments):
##            for j in range(self.num_weights):
##                self.v[i][j] = np.random.uniform(0, 10, size=(self.patients.shape[1]))
##                self.v[i][j] = self.v[i][j] / np.linalg.norm(self.v[i][j])

        self.dataset = self.generate_dataset( self.num_treatments)

##    def normalize_data(self, patient_features):
##        x = (patient_features - np.min(patient_features, axis=0)) / (
##                np.max(patient_features, axis=0) - np.min(patient_features, axis=0))
##
##        for i in range(x.shape[0]):
##            x[i] = x[i] / np.linalg.norm(x[i])
##
##        return x

    def generate_dataset(self, num_treatments):
        tcga_dataset = dict()
        tcga_dataset['x'] = []
        tcga_dataset['y'] = []
        tcga_dataset['t'] = []
        tcga_dataset['d'] = []
        tcga_dataset['metadata'] = dict()
        #tcga_dataset['metadata']['v'] = self.v
        tcga_dataset['metadata']['treatment_selection_bias'] = self.treatment_selection_bias
        tcga_dataset['metadata']['dosage_selection_bias'] = self.dosage_selection_bias
        tcga_dataset['metadata']['noise_std'] = self.noise_std
        tcga_dataset['metadata']['scaling_parameter'] = self.scaling_parameteter
        

##        for patient in patient_features:
##            t, dosage, y = generate_patient(x=patient, v=self.v, num_treatments=num_treatments,
##                                            treatment_selection_bias=self.treatment_selection_bias,
##                                            dosage_selection_bias=self.dosage_selection_bias,
##                                            scaling_parameter=self.scaling_parameteter,
##                                            noise_std=self.noise_std)
##            tcga_dataset['x'].append(patient)
##            tcga_dataset['t'].append(t)
##            tcga_dataset['d'].append(dosage)
##            tcga_dataset['y'].append(y)

        

        x, t, d, y, ids, ps, response_data = load_data(self.datasetname)
        ps = scale_bias(ps, bias_level)

        for key in ['x', 't', 'd', 'y', 'ids','ps']:
            tcga_dataset[key] = {'x':x, 'y':y, 't':t, 'd':d, 'ids':ids,'ps':ps}[key]

        tcga_dataset['metadata']['y_min'] = np.min(tcga_dataset['y'])
        tcga_dataset['metadata']['y_max'] = np.max(tcga_dataset['y'])

        tcga_dataset['y_normalized'] = (tcga_dataset['y'] - np.min(tcga_dataset['y'])) / (
                np.max(tcga_dataset['y']) - np.min(tcga_dataset['y']))

        train_indices, validation_indices, test_indices = get_split_indices(num_patients=tcga_dataset['x'].shape[0],
                                                                            patients=tcga_dataset['x'],
                                                                            treatments=tcga_dataset['t'],
                                                                            validation_fraction=self.validation_fraction,
                                                                            test_fraction=self.test_fraction)

        tcga_dataset['metadata']['train_index'] = train_indices
        tcga_dataset['metadata']['val_index'] = validation_indices
        tcga_dataset['metadata']['test_index'] = test_indices

        return tcga_dataset, response_data


def get_dataset_splits(dataset):
    dataset_keys = ['x', 't', 'd', 'y','ps', 'y_normalized', 'ids']

    train_index = dataset['metadata']['train_index']
    val_index = dataset['metadata']['val_index']
    test_index = dataset['metadata']['test_index']

    dataset_train = dict()
    dataset_val = dict()
    dataset_test = dict()
    for key in dataset_keys:
        dataset_train[key] = dataset[key][train_index]
        dataset_val[key] = dataset[key][val_index]
        dataset_test[key] = dataset[key][test_index]

    dataset_train['metadata'] = dataset['metadata']
    dataset_val['metadata'] = dataset['metadata']
    dataset_test['metadata'] = dataset['metadata']

    return dataset_train, dataset_val, dataset_test


    
def export_result(out_path, Mise, DPE, PE, lr=0.1, h_dim=1, num_dosage_sample=1,\
                  epoch=1):
    row = 'lr: ' + str(lr) + '_epoch: ' + str(epoch) + '_h_dim: ' + str(h_dim) + 'num_dosage_sample: ' + str(num_dosage_sample) + ' -- '
    row += 'MISE: (' + str(np.mean(Mise)) + ', ' + str(np.std(Mise)) + '), '
    row += 'DPE; (' + str(np.mean(DPE)) + ', ' + str(np.std(DPE)) + '), '
    row += 'PE; (' + str(np.mean(PE)) + ', ' + str(np.std(PE)) + ')\n'
    file = open(out_path, 'a')
    file.write(row)
    file.close()


if __name__ == "__main__":

    args = init_arg()

    dataset_params = dict()
    dataset_params['num_treatments'] = args.num_treatments
    dataset_params['treatment_selection_bias'] = args.treatment_selection_bias
    dataset_params['dosage_selection_bias'] = args.dosage_selection_bias
    dataset_params['save_dataset'] = args.save_dataset
    dataset_params['validation_fraction'] = args.validation_fraction
    dataset_params['test_fraction'] = args.test_fraction


    replications = 5
    bias_level = 0.2 # base level = 0
    dataset_name = 'mimiciv_mv'
    out_path = './SCIGAN_' + dataset_name+ '_bias_level_' + str(int(bias_level*100)) + '.txt'
    file = open(out_path, 'w')
    file.write('')
    file.close()
    hyperparameters = {'mimiciii_mv':{'h_dims':[46], 'lrs':[0.001], \
                                'num_dosage_samples':[5],\
                                'epochs':[ 150,200]},\
                       'mimiciv_mv':{'h_dims':[34 ], 'lrs':[0.001], \
                                'num_dosage_samples':[5],\
                                'epochs':[ 50,100 ]},\
                       'TCGA_m6':{'h_dims':[24,34,44], 'lrs':[0.001], \
                                'num_dosage_samples':[4,5,6],\
                                'epochs':[1000, 1200, 2000, 2200, 3000]},\
                       'mimic_vaso_m6':{'h_dims':[38,50,72], 'lrs':[0.001], \
                                'num_dosage_samples':[4,5,6],\
                                'epochs':[1000, 1200, 2000, 2200, 3000]},\
                       'USCMR_m6':{'h_dims':[10,15,20], 'lrs':[0.001], \
                                'num_dosage_samples':[5],\
                                'epochs':[1000]},\
                       'mimiciv_seda':{'h_dims':[28 ], 'lrs':[0.001], \
                                'num_dosage_samples':[5],\
                                'epochs':[1000 ]},\
                       'mimiciii_seda':{'h_dims':[26], 'lrs':[0.001], \
                                'num_dosage_samples':[6],\
                                'epochs':[1000]},\
                       'mimic_sed':{'h_dims':[26], 'lrs':[0.001], \
                                'num_dosage_samples':[4],\
                                'epochs':[3000]},\
                       'mimiciv_coag':{'h_dims':[34,38,42], 'lrs':[0.001], \
                                'num_dosage_samples':[4,5,6],\
                                'epochs':[1000, 1200, 2000, 2200, 3000]},\
                       'mimiciii_seda1':{'h_dims':[26,30,34], 'lrs':[0.001], \
                                'num_dosage_samples':[4,5,6],\
                                'epochs':[1000, 1200, 2000, 2200, 3000]},\
                       'eicu_mv_m6':{'h_dims':[50,60,70], 'lrs':[0.001], \
                                'num_dosage_samples':[4,5,6],\
                                'epochs':[1000, 1200, 2000, 2200, 3000]}}[dataset_name]


    hidden_dims = hyperparameters['h_dims']
    lrs = hyperparameters['lrs']
    num_dosage_samples = hyperparameters['num_dosage_samples']
    epochs = hyperparameters['epochs']
    for lr in lrs:
        for h_dim in hidden_dims:
            for num_dosage_sample in num_dosage_samples:
                for epoch in epochs:
                    np.random.seed(10)
                    tf.random.set_random_seed(10)
                    Mise, DPE, PE = [], [], []
                    for rep in range(replications):
    

                        data_class = TCGA_Data(dataset_params, dataset_name)
                        dataset, response_data = data_class.dataset
                        dataset_train, dataset_val, dataset_test = get_dataset_splits(dataset)

                        export_dir = 'saved_models/' + args.model_name
                        if os.path.exists(export_dir):
                            shutil.rmtree(export_dir)

                        params = {'num_features': dataset_train['x'].shape[1], 'num_treatments': args.num_treatments,
              'num_dosage_samples': num_dosage_sample, 'export_dir': export_dir,
              'alpha': args.alpha, 'batch_size': args.batch_size, 'h_dim': h_dim,
              'h_inv_eqv_dim': h_dim, 'epochs': epoch, 'lr':lr}

                        model_baseline = SCIGAN_Model(params, dataset_name=dataset_name)

                        model_baseline.train(Train_X=dataset_train['x'], Train_T=dataset_train['t'], Train_D=dataset_train['d'],
                         Train_Y=dataset_train['y_normalized'], Train_ps=dataset_train['ps'],verbose=args.verbose)

                        mise, dpe, pe = compute_eval_metrics(dataset, dataset_test['x'], dataset_test['ids'],response_data, num_treatments=params['num_treatments'],
                                         num_dosage_samples=params['num_dosage_samples'], model_folder=export_dir,dataset_name=dataset_name)
                        Mise.append(mise)
                        DPE.append(dpe)
                        PE.append(pe)

                    export_result(out_path, Mise, DPE, PE, lr=lr, h_dim=h_dim, \
                                 num_dosage_sample=num_dosage_sample, epoch=epoch)
