import numpy as np
from scipy.optimize import minimize
from scipy.integrate import romb


def sigmoid(x):
    return 1 / (1 + np.exp(-x))




class idx_generator:
    def __init__(self, size, bs):
        self.idx = np.arange(0, size)
        self.bs = bs
        

    def sample(self):
        return np.random.choice(self.idx, size=self.bs)


def data_split(x,t,d,y,ids, test_ratio, num_treatments=1):
    n = len(d)
    idx = np.arange(n)
    np.random.shuffle(idx)
    train_size = int(n * (1 - test_ratio))
    propensity = []
    for i in range(num_treatments):
        propensity.append(1 / num_treatments)
    data_tr = {'x':[], 't':[],'d':[],'y':[],'propensity':[],'ids':[]}
    data_te = {'x':[], 't':[],'d':[],'y':[],'ids':[]}
    for i in idx[:train_size]:
        data_tr['x'].append(x[i])
        data_tr['t'].append(t[i])
        data_tr['d'].append(d[i])
        data_tr['y'].append(y[i])
        data_tr['ids'].append(ids[i])
        data_tr['propensity'].append(propensity) # initialize propensities

    for i in idx[train_size:]:
        data_te['x'].append(x[i])
        data_te['t'].append(t[i])
        data_te['d'].append(d[i])
        data_te['y'].append(y[i])
        data_te['ids'].append(ids[i])


    for i in data_tr:
        data_tr[i] = np.array(data_tr[i])
    for i in data_te:
        data_te[i] = np.array(data_te[i])

    return data_tr, data_te


    
def cal_mse(x, y):
    return np.sum(np.square(x - y))


def get_train_samples(data_tr, idx):
    x = data_tr['x'][idx]
    t = data_tr['t'][idx]
    d = data_tr['d'][idx]
    y = data_tr['y'][idx]
    propensity = data_tr['propensity'][idx]
    return [x,t,d], [propensity, y]


def save_weights(model, file_path):
    weights = dict([(str(i), weight)
                        for i, weight in enumerate(model.get_weights())])
    np.savez(file_path, **weights)


def load_weights(file_path):
    weights = np.load(file_path)
    num_weights = len(weights.files)
    weight_list = [weights[str(idx) + ".npy"] for idx in range(num_weights)]
    return weight_list

def early_stop(current, best, tol):
    if (current - best) > tol:
        return True
    else:
        return False


def predict(model, data_tr, bs, train_size):
    propensity = []
    pre_y = []
    a = 0
    b = a + bs
    all_idx = np.arange(0, train_size)
    while a < train_size:
        idx = all_idx[a:b]
        x = data_tr['x'][idx]
        t = data_tr['t'][idx]
        d = data_tr['d'][idx]
        propensity_t, pre_y_t = model.predict([x,t,d])
        if propensity == []:
            propensity = propensity_t
            pre_y = pre_y_t
        else:
            propensity = np.concatenate((propensity, propensity_t), axis=0)
            pre_y = np.concatenate((pre_y, pre_y_t), axis=0)
        a += bs
        b += bs
    return propensity, pre_y.flatten()


##def get_patient_outcome(x, v, _, t, scaling_parameter=10):
##    mu = 4 * (t-0.5)**2*np.sin(np.pi/2*t) * 2 * \
##             ((sum(v[1]*x) / sum(v[2]*x))**0.5 + 10 * sum(v[0]*x))
##
##    return mu
##
##
##def get_true_dose_response_curve(v, patient, treatment_idx):
##    def true_dose_response_curve(dosage):
##        y = get_patient_outcome(patient, v, treatment_idx, dosage)
##        return y
##
##    return true_dose_response_curve


    


def evaluate_model(model, data, response, num_treatments, dosage_samples, dataset):
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
    patients = data['x']
    ids = data['ids']


   ####
    test_data = dict()
    test_data['x'] = np.repeat(np.expand_dims([-0.0013521685087687012,-0.01553670947552708,-0.01579938616009668,-0.01553670947552708,-0.01579938616009668,-0.01579938616009668,-0.01579938616009668,0.01054708530223416,-0.012384589260691886,-0.015234631288272042,-0.00870711567671749,-0.01548417413861316,0.010205605612293682,-0.008851587853230772,-0.013514099004341164,0.023470778183058465,-0.014643608747990443,0.006291723012206646,-0.015379103464785321,-0.011228811848585648,0.02031865796822327,-0.008181762307578291,-0.00883845401900229,0.005214748605471289,0.009942928927724084,0.0015372750214968948,0.0039013651826232914,-0.010545852468704688,-0.006080348831021495,0.009942928927724084,262.660622506755]
, axis=0), num_integration_samples, axis=0)
    test_data['t'] = np.repeat(0, num_integration_samples)
    test_data['d'] = treatment_strengths
    _, pred_dose_response = predict(model, test_data, \
                                            num_integration_samples,\
                                            num_integration_samples)
    print('rep y',pred_dose_response.tolist())
 

    
    for patient, p_id in zip(patients, ids):
        for treatment_idx in range(num_treatments):
            test_data = dict()
            test_data['x'] = np.repeat(np.expand_dims(patient, axis=0), num_integration_samples, axis=0)
            test_data['t'] = np.repeat(treatment_idx, num_integration_samples)
            test_data['d'] = treatment_strengths

            _, pred_dose_response = predict(model, test_data, \
                                            num_integration_samples,\
                                            num_integration_samples)
##            true_outcomes = [get_patient_outcome(patient, v, treatment_idx, d) for d in
##                                 treatment_strengths]
            true_outcomes = response[p_id]
            mise = romb(np.square(true_outcomes - pred_dose_response), dx=step_size)
            mises.append(mise)
##            best_encountered_x = treatment_strengths[np.argmax(pred_dose_response)]
##
##
##            def pred_dose_response_curve(dosage):
##                test_data = dict()
##                x = np.expand_dims(patient, axis=0)
##                t = np.expand_dims(treatment_idx, axis=0)
##                d = np.expand_dims(dosage, axis=0)
##
##                _, ret_val = model.predict([x,t,d])
##                return ret_val.flatten()
##
##            true_dose_response_curve = get_true_dose_response_curve(v, \
##                                                                    patient, \
##                                                                    treatment_idx)
##            min_pred_opt = minimize(lambda x: -1 * pred_dose_response_curve(x),
##                                        x0=[best_encountered_x], method="SLSQP", bounds=[(0, 1)])
##            max_pred_opt_y = - min_pred_opt.fun
##            max_pred_dosage = min_pred_opt.x
##            max_pred_y = true_dose_response_curve(max_pred_dosage)
##            min_true_opt = minimize(lambda x: -1 * true_dose_response_curve(x),
##                                        x0=[0.5], method="SLSQP", bounds=[(0, 1)])
##            max_true_y = - min_true_opt.fun
##            max_true_dosage = min_true_opt.x
##
##            dosage_policy_error = (max_true_y - max_pred_y) ** 2
##            dosage_policy_errors.append(dosage_policy_error)
##
##            pred_best.append(max_pred_opt_y)
##            pred_vals.append(max_pred_y)
##            true_best.append(max_true_y)
##        selected_t_pred = np.argmax(pred_vals[-num_treatments:])
##        selected_val = pred_best[-num_treatments:][selected_t_pred]
##        selected_t_optimal = np.argmax(true_best[-num_treatments:])
##        optimal_val = true_best[-num_treatments:][selected_t_optimal]
##        policy_error = (optimal_val - selected_val) ** 2
##        policy_errors.append(policy_error)
    return np.sqrt(np.mean(mises)), 0., 0.#np.sqrt(np.mean(dosage_policy_errors)), np.sqrt(np.mean(policy_errors))
