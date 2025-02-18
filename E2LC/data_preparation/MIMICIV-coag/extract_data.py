'''
The sedation score is according to Riker Sedation-Agitation Scale
'''
import numpy as np
import csv
import matplotlib.pyplot as plt
import math



def read_data(path, constrains = [0,0.9,18,160]):
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
    #for i in range(2, data1.shape[1]-1):
    #    data1 = standardize(np.array(data1), col=[i])
    return data1




def standardize(data, col=None):
    # column 2 = t, last column = i
    if col:
        for i in col:
            mean = np.mean(data[:,i])
            std = np.std(data[:,i])
            data[:,i] = (data[:,i] - mean) / std
    else:
        data = np.array(data)
        print(('y mean',np.mean(data[:, 0]), 'y std',np.std(data[:, 0])))
        data[:, 0] = (data[:, 0] - np.mean(data[:, 0])) / np.std(data[:, 0])
        data[:, 2:-1] = (data[:, 2:-1] - np.mean(data[:, 2:-1]))\
                        / np.std(data[:, 2:-1])
    return data.tolist()

def write_csv(outpath, data):
    with open(outpath, 'w',newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(data)



if __name__ == '__main__':
    path = './mimiciv_coag_raw.csv'
    all_data = read_data(path)
    
    

    ###  real response ###
    x,y,t = [],[],[]
    for i in all_data:
        y.append(i[0])
        x.append(i[1])

    #######

       

    max_t,min_t = max(x), min(x)

    for j,i in enumerate(all_data):
        i[-1] = j # id
        i[1] = (i[1] - min_t) / (max_t - min_t) # normalize dosage to [0,1]
 

    outpath = './mimiciv_coag.csv' # y and features are standardize to N(0,1)
    all_data = standardize(all_data)
    write_csv(outpath, all_data)
