import numpy as np
from scipy.stats import bernoulli
from scipy.stats import norm
import random

import csv



def sigmoid(x):
    return 1 / (1 + np.exp(-x))






def generate_data(n, dz, nf_cont, nf_disc, x_cont_cof, x_disc_cof, \
                         y_cof, t_cof, selection_bias, noise_scale=0.01):
    result = []
    for i in range(n):
        z = [norm.rvs(1, 0.5) for j in range(dz)]
        z = np.array(z)
        x_mu_disc = bernoulli(sigmoid((z * x_disc_cof)[0])).rvs()
        x_loc = np.array((z ** 3 * x_cont_cof))[0]
        eta = np.array([norm.rvs(0, noise_scale) for j in range(nf_cont)])
        x = (x_loc + eta).tolist()


        t_loc = sigmoid((z * t_cof)[0] + selection_bias)
        t = bernoulli(t_loc).rvs()

        mu0 = float(0.8 * sum(((z - 0.5) ** 2 * y_cof)[0] + 1 - 2.2 * sum(z)/5))
        mu1 = float(0.8 * sum(((z - 0.5) ** 2 * y_cof)[0] + 1))
        
        y0 = norm.rvs(mu0, 0.02)
        y1 = norm.rvs(mu1, 0.02)

        if t == 1:
            yfac = y1
            ycf = y0
        else:
            yfac = y0
            ycf = y1
        
        row = [t, yfac, ycf, mu0, mu1] # col = [t, yfac, ycf, mu0, mu1, x]
        row.extend(x_mu_disc)
        row.extend(x)
        row.extend(z.tolist())
        result.append(row)

    return result




def export_csv(data, name):
    path = './source' + name + '.csv'
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter = ',')
        for  i in data:
            writer.writerow(i)

            


if __name__ == '__main__':
    # control randomness
    random.seed(3)
    np.random.seed(3)
    ######
    dz = 8 # dimension of z
    nf_cont = 10 # dimension of x
    nf_disc = 3 # dimension of x
    n = 1000 # num of samples
    x_cont_cof = np.matrix(np.random.normal(np.zeros((dz, nf_cont)), np.ones((dz, nf_cont))))
    x_disc_cof = np.matrix(np.random.normal(np.zeros((dz, nf_disc)), np.ones((dz, nf_disc))))
    y_cof = np.matrix(np.random.normal(np.zeros((dz, 1)), np.ones((dz, 1))))
    t_cof = np.matrix(np.random.normal(np.zeros((dz, 1)), np.ones((dz, 1))))
    selection_bias = -1
    data = generate_data(n, dz, nf_cont, nf_disc, x_cont_cof, x_disc_cof, \
                         y_cof, t_cof, selection_bias)
    
    
    export_csv(data, 'synthetic_combined')
    
    
