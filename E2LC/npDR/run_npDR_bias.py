import numpy as np
import scipy.stats
from npDoseResponseDR import DRCurve
from npDoseResponseDerivDR import DRDerivCurve, NeurNet
from sklearn.neural_network import MLPRegressor
import argparse
from utils import *
import bisect



def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_fraction", default=0.2, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_unit", default=60, type=int)
    parser.add_argument("--epochs", default=500, type=int) # 3 for mimiciii-mv, 30 for mimiciv-seda
    parser.add_argument("--lr", default=0.001, type=float)

    
    
    return parser.parse_args()


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



def data_split(x,t,y,ids,ps, test_ratio, num_treatments=1):
    n = len(t)
    idx = np.arange(n)
    ps_tr = []
    np.random.shuffle(idx)
    train_size = int(n * (1 - test_ratio))
    propensity = []
    data_tr = {'tx':[],'y':[], 'ids':[]}
    data_te = {'tx':[],'y':[], 'ids':[]}
    for i in idx[:train_size]:
        temp = [t[i]]
        temp.extend(x[i].tolist())
        data_tr['tx'].append(temp)
        data_tr['y'].append(y[i])
        data_tr['ids'].append(ids[i])
        ps_tr.append(ps[i])

    for i in idx[train_size:]:
        temp = [t[i]]
        temp.extend(x[i].tolist())
        data_te['tx'].append(temp)
        data_te['y'].append(y[i])
        data_te['ids'].append(ids[i])

    for key in data_tr:
        data_tr[key] = np.array(data_tr[key])
        data_te[key] = np.array(data_te[key])
        

    return data_tr, data_te, ps_tr


def scale_bias(ps, bias_level):
    ps = np.array(ps) ** bias_level
    ps = ps / sum(ps)
    return ps.tolist()

def replicate(data_tr, ps_tr, max_rep=6):
    thresholds = np.linspace(0,1,max_rep+1).tolist()
    result = {'tx':[],'y':[], 'ids':[]}
    ps_norm = np.array(ps_tr)
    ps_norm = (ps_norm - min(ps_norm)) / (max(ps_norm) - min(ps_norm))
    ps_norm = ps_norm.tolist()
    for idx, ps in enumerate(ps_norm):
        reps = min(bisect.bisect_right(thresholds, ps), max_rep)
        tx = [data_tr['tx'][idx]] * reps
        y = [data_tr['y'][idx]] * reps
        ids = [data_tr['ids'][idx]] * reps
        result['tx'].extend(tx)
        result['y'].extend(y)
        result['ids'].extend(ids)
    for key in result:
        result[key] = np.array(result[key])
    return result
        
    

if __name__ == "__main__":

    args = init_arg()


    #num_epoch = args.epochs


    MSE = []

    dataset = 'mimiciv_coag'
    bias_level = 0.2 # base level = 0
    out_path = './npDR_' + dataset + '_bias_level_' + str(int(bias_level*100)) + '.txt'
    file = open(out_path, 'w')
    file.write('')
    file.close()
    test_ratio = 0.2
    batch_size = 150
    hyperparameters = {'mimiciii_seda':{'num_units':[1.3], 'lrs':[0.0001],\
                                'layers':[1]},
                       'mimiciv_seda':{'num_units':[1.1], 'lrs':[0.001],\
                                'layers':[3]},
                       'mimiciii_mv':{'num_units':[1.3], 'lrs':[0.001],\
                                'layers':[2]},
                       'mimiciv_mv':{'num_units':[1.1], 'lrs':[0.01],\
                                'layers':[1]},
                       'mimiciv_coag':{'num_units':[1.3], 'lrs':[0.00001],\
                                'layers':[1]}}[dataset]
    replications = 5
    with open('../data/' + dataset + '_response_curve_calibrate.pickle', 'rb') as file:
        response_data = pickle.load(file)
    
    for lr in hyperparameters['lrs']:
        for num_unit in hyperparameters['num_units']:
            for layer in hyperparameters['layers']:
                np.random.seed(3)
                Mise = []
            
                for rep in range(replications):
                    x,t,y,ids,ps = load_data(dataset)
                    dx = x.shape[1]
                    hidden = int(dx * num_unit)
                    hidden_layer_sizes = tuple([hidden] * layer)
                    data_tr, data_te, ps_tr = data_split(x,t,y,ids, ps,test_ratio)
                    ps_tr = scale_bias(ps_tr, bias_level)
                    data_tr = replicate(data_tr, ps_tr)
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
                h=h, kern="epanechnikov", h_cond=None, self_norm=True, print_bw=False)
                    mise = evaluate_model(curve, data_te, response_data)
                    Mise.append(mise)
                if len(Mise) == replications:
                    export_result(Mise, out_path, lr, num_unit, layer)
            
            
