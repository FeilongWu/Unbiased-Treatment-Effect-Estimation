'''
The sedation score is according to Riker Sedation-Agitation Scale
'''
import numpy as np
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import bisect

def standardize(data, col=[0]):
    for i in col:
        mean = np.mean(data[:,i])
        std = np.std(data[:,i])
        data[:,i] = (data[:,i] - mean) / std
    return data

def normalize(data, col=[1]):
    for i in col:
        Min = np.min(data[:,i])
        Max = np.max(data[:,i])
        data[:,i] = (data[:,i] - Min) / (Max - Min)
    print(('treatment min: ', Min, ' max - min: ', Max - Min))
    return data

def read_data(path, constrains = [0.0,1.25,0,10]):
    # min_t, max_t, min_y, max_y = constrains
    data = []
    with open(path) as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            temp = []
            for i in row:
                temp.append(float(i))
            data.append(temp)
    min_t, max_t, min_y, max_y = constrains
    data1 = []
    count = 0
    for i in data:
        y,t = i[:2]
        if (min_t <= t) and (t <= max_t) and (min_y <= y) and (y <= max_y):
            data1.append(i)
            data1[-1].append(count)
            count += 1
    data1 = np.array(data1)
    for i in range(2, data1.shape[1]-1):
        data1 = standardize(data1, col=[i])
    return data1.tolist()
                
            


def write_csv(outpath, data):
    with open(outpath, 'w',newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(data)




if __name__ == '__main__':

    path = './mimiciii_seda_raw.csv'
    all_data = read_data(path)
    
    

    
    #### density  ####
    x,y,t = [],[],[]
    for i in all_data:
        y.append(i[0])
        x.append(i[1])

    ####

    
    threshold = 3.0 # label = 1 if y <= 3 being sedated
    max_t,min_t = max(x), min(x)

    
    for j,i in enumerate(all_data):
        if i[0] <= threshold:
            i[0] = 1
        else:
            i[0] = 0
        i[-1] = j # id
        i[1] = (i[1] - min_t) / (max_t - min_t) # normalize dosage to [0,1]
    outpath = './mimiciii_seda.csv'
    write_csv(outpath, all_data)
