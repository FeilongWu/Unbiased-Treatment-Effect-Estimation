import json
import numpy as np
import csv


def convert_static(static):
    # static = [age, sex, race, cancer, diabete, height]
    sex = [0, 0]
    race = [0, 0, 0, 0, 0]
    cancer = [0, 0]
    diabete = [0, 0]

    result = [static[0]]
    sex[static[1] - 1] = 1
    race[static[2] - 1] = 1
    cancer[static[3] - 1] = 1
    #diabete[static[4] - 1] = 1
    
    result.extend(sex)
    result.extend(race)
    result.extend(cancer)
    #result.extend(diabete)
    result.append(static[-1])

    return result


if __name__ == '__main__':
    path = './ventilation_duration_30.json'
    file = open(path,'r')
    data = json.load(file)
    file.close()
    n = len(data) # total = 1928
    num_instance = int(n / 2) # select half of the population
    table = []
    indx = np.random.permutation(n)[:num_instance]
    keys = list(data.keys())
    for i in indx:
        i = keys[i]
        static = convert_static(data[i]['static'])
        for icu_stay in data[i]['dynamic']:
            for j in data[i]['dynamic'][icu_stay]['x'][:10]:
                temp = []
                temp.extend(static)
                temp.extend(j)
                table.append(temp)
    file = open('./mimic_extract_x.csv', 'w')
    file.close()
    with open('./mimic_extract_x.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in table:
            writer.writerow(i)
    
    
