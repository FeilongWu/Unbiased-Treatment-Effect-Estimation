'''
This programs export extracted data from eICU datasets to CSV files by hospital.
'''

import json
import csv

def read_data(path):
    file = open(path)
    data = file.read()
    data = data.replace("'", '"')
    data = json.loads(data)
    return data


def export_csv(hosp, data):
    # hosp: [(size, hospital_id),...]
    # data: {hospital_id: [x,t,y]}

    for i in hosp:
        hid = i[1]
        table = data[hid]
        path = './' + str(hid)+ '.csv'
        with open(path, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            for row in table:
                temp = [row[-2], row[-1]] # [T, y]
                race = [0, 0, 0, 0, 0, 0]
                race[row[2] - 1] = 1
                temp.extend(race)
                temp.append(row[1] - 1) # sex
                temp.append(row[0]) # age
                temp.extend(row[3:-2])
                writer.writerow(temp)


if __name__ == '__main__':
    path = 'C://Users//feilo//Desktop//extract_eICU//vasopressor_hospital.txt'
    data = read_data(path)
    num_hosp = 12 # the number of hospitals with the greatest amount of data
    hosp_size = []
    for i in data:
        size = len(data[i])
        hosp_size.append((size, i))
    hosp_size = sorted(hosp_size, reverse=True) # sort the data size largest to least
    export_csv(hosp_size[:num_hosp], data)
    
