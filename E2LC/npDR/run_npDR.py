import numpy as np
import scipy.stats
from npDoseResponseDR import DRCurve
from npDoseResponseDerivDR import DRDerivCurve, NeurNet
from sklearn.neural_network import MLPRegressor
import argparse
from utils import *



def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_fraction", default=0.2, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_unit", default=60, type=int)
    parser.add_argument("--epochs", default=500, type=int) # 3 for mimiciii-mv, 30 for mimiciv-seda
    parser.add_argument("--lr", default=0.001, type=float)

    
    
    return parser.parse_args()



if __name__ == "__main__":

    args = init_arg()


    #num_epoch = args.epochs


    MSE = []

    dataset = 'mimiciv_coag'
    out_path = './npDR_' + dataset + '.txt'
    file = open(out_path, 'w')
    file.write('')
    file.close()
    test_ratio = 0.2
    batch_size = 150
    hyperparameters = {'mimiciii_seda':{'num_units':[1.3], 'lrs':[0.0001],\
                                'layers':[1]},
                       'mimiciv_seda':{'num_units':[1.1,1.2,1.3], 'lrs':[0.01,0.001,0.0001],\
                                'layers':[1,2,3]},
                       'mimiciii_mv':{'num_units':[1.1], 'lrs':[0.01],\
                                'layers':[1]},
                       'mimiciv_mv':{'num_units':[1.1,1.2,1.3], 'lrs':[0.01,0.001,0.0001],\
                                'layers':[1,2,3]},
                       'mimiciv_coag':{'num_units':[1.3], 'lrs':[0.00001],\
                                'layers':[1]}}[dataset]
    replications = 5
    
    for lr in hyperparameters['lrs']:
        for num_unit in hyperparameters['num_units']:
            for layer in hyperparameters['layers']:
                np.random.seed(3)
                Mise = []
            
                for rep in range(replications):
                    x,t,y,ids,response_data = load_data(dataset)
                    dx = x.shape[1]
                    hidden = int(dx * num_unit)
                    hidden_layer_sizes = tuple([hidden] * layer)
                    data_tr, data_te = data_split(x,t,y,ids, test_ratio)
                    regr_nn2 = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation='relu', learning_rate='adaptive', 
                        learning_rate_init=lr, random_state=3, max_iter=500)
                    reg_mod = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation='relu', learning_rate='adaptive', 
                       learning_rate_init=lr, random_state=3, max_iter=500)
                    n = len(data_tr['y'])
                    nu = np.random.randn(n)
                    theta = 1/(np.linspace(1, dx, dx)**2)
            
                    T_sim = scipy.stats.norm.cdf(3*np.dot(data_tr['tx'][:,1:], theta)) + 3*nu/4 - 1/2
                    h = 4*np.std(T_sim)*n**(-1/5)
            #m_est_dr5, sd_est_dr5 = DRCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est="DR", mu=reg_mod, 
            #    condTS_type='kde', condTS_mod=regr_nn2, tau=0.001, L=5, 
            #    h=h, kern="epanechnikov", h_cond=None, self_norm=True, print_bw=True)
            # Y_sim: 1D array
            # X_dat: batch_size * (1+dx), first column is t
            # t_qry: 1D array, t for testing
                    Y_sim = data_tr['y']
                    X_dat = data_tr['tx']
                    samples_power_of_two = 6
                    num_integration_samples = 2 ** samples_power_of_two + 1
                    step_size = 1. / num_integration_samples
                    t_qry = np.linspace(np.finfo(float).eps, 1, num_integration_samples)
                    curve,_ = DRCurve(Y=Y_sim, X=X_dat, t_eval=t_qry, est="DR", mu=reg_mod, 
                condTS_type='kde', condTS_mod=regr_nn2, tau=0.001, L=5, 
                h=h, kern="epanechnikov", h_cond=None, self_norm=True, print_bw=True)
                    print(curve.tolist())
                    exit(0)
                    mise = evaluate_model(curve, data_te, response_data)
                    Mise.append(mise)
                if len(Mise) == replications:
                    export_result(Mise, out_path, lr, num_unit, layer)
            
            
