from models.admit import *
from args import Helper
from torch.utils.data import DataLoader
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




def load_data(dataset,rep, gamma=None):    
    x = []
    d = []
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
    return x, d, y, ids, response


def export_result(out_path, Mise, num_unit, lr, k):
    row = 'lr: ' + str(lr) + '_num_unit: ' + str(num_unit) + '_k: ' + str(k) + ' -- '
    row += 'MISE: (' + str(np.mean(Mise)) + ', ' + str(np.std(Mise)) + ')\n'
    file = open(out_path, 'a')
    file.write(row)
    file.close()


def main(dataset='mimic', gamma=None):
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
    out_path = './ADMIT_' + dataset + '.txt'
    file = open(out_path, 'w')
    file.write('')
    file.close()
    #hyperparameters = {'mimic':{'num_units':[40,50,60], 'lrs':[0.001, 0.0001, 0.0002],\
    #                            'k':[4,5,6]},\
    #                   'mimiciii_seda':{'num_units':[38], 'lrs':[0.0003],\
    #                            'k':[6]},\
    #                   'ihdp':{'num_units':[35,45,55], 'lrs':[0.001, 0.0001, 0.0002],\
    #                            'k':[4,5,6]},\
    #                   'mimic':{'num_units':[40,50,60], 'lrs':[0.001, 0.0001, 0.0002],\
    #                            'k':[4,5,6]},\
    #                   'mimiciv_coag':{'num_units':[38], 'lrs':[0.001],\
    #                            'k':[6]}}[dataset]

    hyperparameters = {'synthetic':{'num_units':[10,14,18], 'lrs':[0.001,0.0001,0.00001],\
                                'k':[4,5,6]},
                       'news':{'num_units':[430,530,630], 'lrs':[0.001,0.0001,0.00001],\
                                'k':[4,5,6]},
                       'mimiciii':{'num_units':[30,40,50], 'lrs':[0.001,0.0001,0.00001],\
                                'k':[4,5,6]},
                       'tcga':{'num_units':[460,560,660], 'lrs':[0.001,0.0001,0.00001],\
                                'k':[4,5,6]}}[dataset]
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
                    x,t,y,ids,response_data = load_data(dataset,rep, gamma=gamma)
                    args.input_dim = x.shape[1]
                    if args.scale:
                        args.scaler = StandardScaler().fit(y.reshape(-1,1))
                    data_tr, data_te = data_split(x,t,y,ids, test_ratio)
                    data_tr = DataLoader(createDS(data_tr), batch_size=batch_size, shuffle=True)
                    data_te = DataLoader(createDS(data_te), batch_size=1, shuffle=False)
    
    

                    model = ADMIT(args, h_unit=num_unit, dataset=dataset)
                    model.to(device)

                    model = train(model, data_tr, args, k=k)
                    mise = evaluate_model(model, data_te, response_data, args)
                    Mise.append(mise)
                export_result(out_path, Mise, num_unit, lr, k)
    

if __name__ == "__main__":
    dataset = 'synthetic'
    gamma = None
    main(dataset=dataset, gamma = gamma)
