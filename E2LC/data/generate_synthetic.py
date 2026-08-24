import csv
import numpy as np
import pickle
import torch



def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def generate_data(size, replication):
    torch.manual_seed(3)
    ytx, curves = {}, {}
    samples_power_of_two = 6
    num_integration_samples = 2 ** samples_power_of_two + 1
    step_size = 1. / num_integration_samples
    treatment_strengths = np.linspace(np.finfo(float).eps, 1,
                                      num_integration_samples)
    treatment_strengths = torch.tensor(treatment_strengths.tolist()).unsqueeze(-1)
    for i in range(replication):
        ytx[i] = []
        curves[i] = []
        for j in range(size):
                x = torch.rand(6)
                x1,x2,x3,x4,x5,x6 = x
                t = (10. * torch.sin(torch.max(x1, torch.max(x2, x3))) \
                     + torch.max(x3, torch.max(x4, x5))**3)/(1. + (x1 + x5)**2) \
                     + torch.sin(0.5 * x3) * (1. + torch.exp(x4 - 0.5 * x3)) \
                     + x3**2 + 2. * torch.sin(x4) + 2.*x5 - 6.5
                t = t + torch.randn(1) * 0.5
                t = sigmoid(t)
                y = torch.cos((t-0.5) * 3.14159 * 2.) \
                    * (t**2 + (4.*torch.max(x1, x6)**3)/(1. + 2.*x3**2)*torch.sin(x4))\
                    + torch.randn(1) * 0.5
                
                ytx[i].append(torch.cat((y,
                                         t,
                                         x), 0).tolist())
                curve1 = torch.cos((treatment_strengths-0.5) * 3.14159 * 2.) \
                    * (treatment_strengths**2 + (4.*torch.max(x1, x6)**3)/(1. + 2.*x3**2)*torch.sin(x4))
                curves[i].append(curve1.flatten().tolist())

    return ytx, curves


def to_csv(replication, synthetic_ytx, curves):
    for i in range(replication):
        path = './synthetic_rep' + str(i) + '.csv'
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for row in synthetic_ytx[i]:
                writer.writerow(row)
        path = './synthetic_response_rep' + str(i) + '.csv'
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for row in curves[i]:
                writer.writerow(row)
                
            

size = 5000
replication = 5
synthetic_ytx, curves = generate_data(size, replication)
to_csv(replication, synthetic_ytx, curves)




    
