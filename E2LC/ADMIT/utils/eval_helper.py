import torch
import time
from utils.log_helper import save_obj, load_obj
import numpy as np
from scipy.integrate import romb


def sigmoid(x):
    return 1 / (1 + np.exp(-x))





def evaluate_model(model, data, response, args):
    mises = []
    dosage_policy_errors = []
    policy_errors = []
    pred_best = []
    pred_vals = []
    true_best = []
    samples_power_of_two = 6
    num_integration_samples = 2 ** samples_power_of_two + 1
    step_size = 1. / num_integration_samples
    treatment_strengths = np.linspace(np.finfo(float).eps, 1, num_integration_samples)
 ####
    x = torch.tensor([-0.11540782514034661,0.8242255917447336,-1.5599145275730124,-0.24128650621655545,8.73212459828649,-0.5361902647381805,-0.6497129024645434,1.5119081683189117,-0.3923713880373228,0.03952063450904228,-0.204804665405613,1.7002111008580854,-0.18061950848917854,-0.3239757279657241,-1.0126540685073162,1.478510244675789,1.2163167050119934,-0.5950716120162598,-0.2676432346583849,-0.49156632034094055,-0.7198280672485246,-0.6671920535590271,-0.7728559859759158,0.2223755636965808,-0.5449469723080683,-0.5035694482656016,-1.1175848204209877,0.5662087914790986,0.6330203262212626,0.6068362474940249,-1.0201620106422626,-0.2729462661116843,-0.34611418780661024,-0.48053898873144046,-0.6108766825045011]
).float()
    x = x.repeat(num_integration_samples, 1).to(args.device)
    t = torch.from_numpy(treatment_strengths).float().to(args.device)
    pre_y, _, __ = model.forward(x, t)
    print('pre_y',pre_y.cpu().flatten().tolist())
#####
    for batch in data:
        x1 = batch['x'].float().to(args.device)
        x = x1.repeat(num_integration_samples, 1)
        idx = batch['ids'].float().item()
        t = torch.from_numpy(treatment_strengths).float().to(args.device)
        pre_y, _, __ = model.forward(x, t)
        pred_dose_response = pre_y.flatten().detach().cpu().numpy()
        if args.scale:
            pred_dose_response = args.scaler.inverse_transform(pred_dose_response.reshape(-1, 1)).squeeze()
        
        #test_data = dict()
        patient = x1[0].detach().cpu().numpy()
        #test_data['x'] = np.repeat(np.expand_dims(patient, axis=0), num_integration_samples, axis=0)
        #test_data['d'] = treatment_strengths # dosage

        true_outcomes = response[idx]
        mise = romb(np.square(true_outcomes - pred_dose_response), dx=step_size)
        mises.append(mise)

    return np.sqrt(np.mean(mises))

##def eval(model, args, train_data, eval_data, test_data):
##    if args.data == 'sim':
##        x=torch.rand([10000, 6])
##    model.eval()
##    
##    n_test = args.n_test
##    n_test = 100
##    t_grid_hat = torch.zeros(n_test)
##    t_grid = torch.zeros(n_test)
##    mse_id = torch.zeros(n_test)
##
##    starttime = time.time()
##
##    x = x.to(args.device)
##    for i in range(n_test):
##        t = (torch.ones(x.shape[0]) * test_data.t[i]).to(args.device)
##        out = model(x, t)
##        out = out[0].data.squeeze().cpu()
##
##        if args.scale:
##            out = args.scaler.inverse_transform(out.reshape(-1, 1)).squeeze()
##            out = torch.tensor(out)
##
##        t_grid_hat[i] = out.mean()
##        ture_out = test_data.get_outcome(x.cpu(), t.cpu())
##        t_grid[i] = ture_out.mean()
##        mse_id[i] = ((out - ture_out).squeeze() ** 2).mean()
##
##    estimation = t_grid_hat.cpu().numpy()
##    savet = test_data.t.cpu().numpy()
##    truth = t_grid.cpu().numpy()
##    dir = '../plot/{}/{}/'.format(args.data, 'rwnet')
##    save_obj(estimation, dir, 'esti')
##    save_obj(savet, dir, 't')
##    save_obj(truth, dir, 'truth')
##
##    mse = ((t_grid_hat.squeeze() - t_grid.squeeze()) ** 2).mean().data
##    mse_id = mse_id.mean().data
##    
##    endtime = time.time()
##
##    print('eval time cost {:.3f}'.format(endtime - starttime))
##
##    return t_grid_hat, mse, mse_id
