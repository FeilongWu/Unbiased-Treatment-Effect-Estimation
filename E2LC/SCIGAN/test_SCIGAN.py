import argparse
import os
import shutil
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np

from data_simulation import get_dataset_splits, TCGA_Data
from SCIGAN import SCIGAN_Model
from utils.evaluation_utils import compute_eval_metrics

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
    parser.add_argument("--epochs", default=500, type=int)

    return parser.parse_args()



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


    replications = 2
    dataset_name = 'mimiciv_coag'
    out_path = './SCIGAN_' + dataset_name + '.txt'
    file = open(out_path, 'w')
    file.write('')
    file.close()
    #hyperparameters = {'mimiciii_mv':{'h_dims':[42,46,50], 'lrs':[0.001], \
    #!                            'num_dosage_samples':[4,5,6],\
    #                           'epochs':[1000, 1200, 2000, 2200, 3000]},\
    #                   'mimiciv_mv':{'h_dims':[34,38,42], 'lrs':[0.001], \
    #                            'num_dosage_samples':[4,5,6],\
    #                            'epochs':[1000, 1200, 2000, 2200, 3000]},\
    #                   'TCGA_m6':{'h_dims':[24,34,44], 'lrs':[0.001], \
    #                            'num_dosage_samples':[4,5,6],\
    #                            'epochs':[1000, 1200, 2000, 2200, 3000]},\
    #                   'mimic_vaso_m6':{'h_dims':[38,50,72], 'lrs':[0.001], \
    #                            'num_dosage_samples':[4,5,6],\
    #                            'epochs':[1000, 1200, 2000, 2200, 3000]},\
    #                   'USCMR_m6':{'h_dims':[10,15,20], 'lrs':[0.001], \
    #                            'num_dosage_samples':[4,5,6],\
    #                            'epochs':[1000, 1200, 2000, 2200, 3000]},\
    #                   'mimiciv_seda':{'h_dims':[28,32,36], 'lrs':[0.001], \
    #                            'num_dosage_samples':[4,5,6],\
    #                            'epochs':[1000, 1200, 2000, 2200, 3000]},\
    #                   'mimiciii_seda':{'h_dims':[26,30,34], 'lrs':[0.001], \
    #                            'num_dosage_samples':[4,5,6],\
    #                            'epochs':[1000, 1200, 2000, 2200, 3000]},\
    #                   'mimic_seda1':{'h_dims':[26], 'lrs':[0.001], \
    #                            'num_dosage_samples':[4],\
    #                            'epochs':[3000]},\
    #                   'mimiciv_coag':{'h_dims':[34,38,42], 'lrs':[0.001], \
    #                            'num_dosage_samples':[4,5,6],\
    #                            'epochs':[1000, 1200, 2000, 2200, 3000]},\
    #                   'mimic_diur_m6':{'h_dims':[46,56,66], 'lrs':[0.001], \
    #                            'num_dosage_samples':[4,5,6],\
    #                            'epochs':[1000, 1200, 2000, 2200, 3000]},\
    #                   'eicu_mv_m6':{'h_dims':[50,60,70], 'lrs':[0.001], \
    #                            'num_dosage_samples':[4,5,6],\
    #                            'epochs':[1000, 1200, 2000, 2200, 3000]}}[dataset_name]





    # mimiciii_seda -- epoch -- 1000
    hyperparameters = {'mimiciii_mv':{'h_dims':[46], 'lrs':[0.001], \
                                'num_dosage_samples':[5],\
                                'epochs':[ 1200]},\
                       'mimiciv_mv20':{'h_dims':[34 ], 'lrs':[0.001], \
                                'num_dosage_samples':[5],\
                                'epochs':[ 2200 ]},\
                       'TCGA_m6':{'h_dims':[24,34,44], 'lrs':[0.001], \
                                'num_dosage_samples':[4,5,6],\
                                'epochs':[1000, 1200, 2000, 2200, 3000]},\
                       'mimic_vaso_m6':{'h_dims':[38,50,72], 'lrs':[0.001], \
                                'num_dosage_samples':[4,5,6],\
                                'epochs':[1000, 1200, 2000, 2200, 3000]},\
                       'USCMR_m6':{'h_dims':[10,15,20], 'lrs':[0.001], \
                                'num_dosage_samples':[5],\
                                'epochs':[1000]},\
                       'mimiciv_seda20':{'h_dims':[28 ], 'lrs':[0.001], \
                                'num_dosage_samples':[5],\
                                'epochs':[1000 ]},\
                       'mimiciii_seda':{'h_dims':[26], 'lrs':[0.001], \
                                'num_dosage_samples':[6],\
                                'epochs':[1000]},\
                       'mimic_sed':{'h_dims':[26], 'lrs':[0.001], \
                                'num_dosage_samples':[4],\
                                'epochs':[3000]},\
                       'mimiciv_coag':{'h_dims':[38], 'lrs':[0.001], \
                                'num_dosage_samples':[4 ],\
                                'epochs':[20]},\
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
                         Train_Y=dataset_train['y_normalized'], verbose=args.verbose)

                        mise, dpe, pe = compute_eval_metrics(dataset, dataset_test['x'], dataset_test['ids'],response_data, num_treatments=params['num_treatments'],
                                         num_dosage_samples=params['num_dosage_samples'], model_folder=export_dir,dataset_name=dataset_name)
                        Mise.append(mise)
                        DPE.append(dpe)
                        PE.append(pe)

                    export_result(out_path, Mise, DPE, PE, lr=lr, h_dim=h_dim, \
                                 num_dosage_sample=num_dosage_sample, epoch=epoch)
