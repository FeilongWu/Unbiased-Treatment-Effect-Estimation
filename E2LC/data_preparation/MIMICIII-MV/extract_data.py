
import numpy as np
import csv
import matplotlib.pyplot as plt
import pickle


#[20,300,24,92]
def read_data(path, constrains = [20,300,24,92]):
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
    #file = open('./hypertension.pickle','rb')
    #hypertensions = pickle.load(file)
    #hyper = []
    for idx,i in enumerate(data):
        y,t = i[:2]
        if (min_t <= t) and (t <= max_t) and (min_y <= y) and (y <= max_y):
            data1.append(i)
            data1[-1].append(count)
            count += 1
            #hyper.append(hypertensions[idx])
    data1 = np.array(data1)

    for i in range(2, data1.shape[1]-1):
        data1 = standardize(np.array(data1), col=[i])
    #file = open('./hypertensions1.pickle','wb')
    #pickle.dump(hyper,file)
    return data1


def extract_dose_response(path, threshold=0.03):
    #threshold: outlier threshold
    data = {}
    include_col = [0, 15, 10, 7] #[subject_id, y, t, feature]
    with open(path) as file:
        reader = csv.reader(file, delimiter=',')
        row_idx = 0
        for row in reader:
            if row_idx == 0:
                row_idx = 1
                continue
            stay_id = int(float(row[2]))
            data[stay_id] = []
            for i in include_col:
                data[stay_id].append(float(row[i]))
    # exclude outlier dosage and label
    dosage = []
    outcome = []
    for i in data:
        dosage.append(data[i][2])
        outcome.append(data[i][1])
    dosage = sorted(dosage)
    outcome = sorted(outcome)
    dosage_out = dosage[int(len(dosage)*(1-threshold))]
    y_out = outcome[int(len(outcome)*(1-threshold))]
    
    data1 = {}
    for i in data:
        if data[i][2] < dosage_out and data[i][1] < y_out:
            data1[i] = data[i]
    return data1


def data_merge(label_data, feature_data):
    data = []
    patient=[]
    for i in label_data:
        if i in feature_data:
            data.append(label_data[i][1:])
            data[-1].extend(feature_data[i][1:])
            patient.append(label_data[i][0])
    print(('num of records: ', len(data), 'num of patients: ', len(set(patient))))
    return data


def standardize(data, col=None):
    # column 2 = t, last column = id
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
    path = './mimiciii_mv_raw.csv'
    all_data = read_data(path)
    


    #### density plot ####
    x,y,t = [],[],[]
    for i in all_data:
        y.append(i[0])
        x.append(i[1])
    plt.scatter(x,y,alpha=0.1)
    ####        

    max_t,min_t = max(x), min(x)
    print('max_t',max_t,'min_t',min_t)

    for j,i in enumerate(all_data):
        i[-1] = j # id
        i[1] = (i[1] - min_t) / (max_t - min_t) # normalize dosage to [0,1]
    

    outpath = './mimiciii_mv.csv' # y and features are standardize to N(0,1)
    all_data = standardize(all_data)
    write_csv(outpath, all_data)
