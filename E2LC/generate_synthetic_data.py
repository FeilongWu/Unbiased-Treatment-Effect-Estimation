'''
This implementation follows ADMIT to generate synthetic values.
'''

import csv
import json
import numpy as np



def load_features(dataset, dx, n):
    data = []
    for i in range(n):
        data.append(np.random.normal(size=dx))
    return np.array(data)



def generate_v(dataset_name, dim_x):
    v = []
    for i in range(3):
        ui = np.random.uniform(0.,10., size=dim_x)
        ui = ui / np.linalg.norm(ui, ord=2)
        v.append(ui)
    v = np.array(v)
    # save v
    v1 = v.tolist()
    #result = {'v':v}
    #file = open('./'+dataset_name+'_metadata.json', 'w')
    #json.dump(result, file)
    #file.close()
    return v[0], v[1], v[2]

def normalize(x):
    xmin = np.min(x, axis=0)
    xmax = np.max(x, axis=0)
    result = (x - xmin) / (
                xmax - xmin)
    for i in range(result.shape[0]):
            result[i] = result[i] / np.linalg.norm(result[i])
    return result


def generate_dataste(v1,v2,v3,x1):
    x = normalize(x1)
    gamma = 3
    dataset = []
    for idx,feature in enumerate(x):
        d_star = sum(v3 * feature) / 2 / sum(v2 * feature)
        beta = (gamma - 1) / d_star + 2 - gamma
        t = np.random.beta(gamma, beta)
        #t = np.random.uniform(0,1)
        mu = 4 * (t-0.5)**2*np.sin(np.pi/2*t) * 2 * \
             ((sum(v2*feature) / sum(v3*feature))**0.5 + 10 * sum(v1*feature))
        y = np.random.normal(mu, 0.5)
        row = [y, t]
        row.extend(feature)
        dataset.append(row)
    return dataset


def export_data(dataset, v1,v2,v3, data):
    dataset = dataset 
    metadata = {}
    metadata['v1'] = v1.tolist()
    metadata['v2'] = v2.tolist()
    metadata['v3'] = v3.tolist()

    file = open('./data/' + dataset + '_metadata.json', 'w')
    json.dump(metadata, file)
    file.close()

    file = open('./data/'+dataset+'_simulate.csv', 'w')
    file.close()
    with open('./data/'+dataset+'_simulate.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in data:
            writer.writerow(i) # [y,t,x]

    
    
    


if __name__ == '__main__':
    dataset_name = 'synthetic'
    dx = 20 # dim of x
    n = 1000 # size of data
    x = np.array(load_features(dataset_name, dx, n))
    np.random.seed(3)
    dim_x = x.shape[1]
    v1, v2, v3 = generate_v(dataset_name, dim_x)
    data = generate_dataste(v1, v2, v3, x)
    export_data(dataset_name, v1,v2,v3, data)
    
