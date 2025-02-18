import numpy as np
from scipy.integrate import romb
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

##from data_simulation import get_patient_outcome
from scipy.optimize import minimize



def sigmoid(x):
    return 1 / (1 + np.exp(-x))




def sample_dosages(batch_size, num_treatments, num_dosages):
    dosage_samples = np.random.uniform(0., 1., size=[batch_size, num_treatments, num_dosages])
    return dosage_samples


def get_model_predictions(sess, num_treatments, num_dosage_samples, test_data):
    batch_size = test_data['x'].shape[0]

    treatment_dosage_samples = sample_dosages(batch_size, num_treatments, num_dosage_samples)
    factual_dosage_position = np.random.randint(num_dosage_samples, size=[batch_size])
    treatment_dosage_samples[range(batch_size), test_data['t'], factual_dosage_position] = test_data['d']

    treatment_dosage_mask = np.zeros(shape=[batch_size, num_treatments, num_dosage_samples])
    treatment_dosage_mask[range(batch_size), test_data['t'], factual_dosage_position] = 1

    I_logits = sess.run('inference_outcomes:0',
                        feed_dict={'input_features:0': test_data['x'],
                                   'input_treatment_dosage_samples:0': treatment_dosage_samples})

    Y_pred = np.sum(treatment_dosage_mask * I_logits, axis=(1, 2))

    return Y_pred


def get_true_dose_response_curve(news_dataset, patient, treatment_idx):
    def true_dose_response_curve(dosage):
        y = get_patient_outcome(patient, news_dataset['metadata']['v'], treatment_idx, dosage)
        return y

    return true_dose_response_curve


def compute_eval_metrics(dataset, test_patients, test_ids, response_data, \
                         num_treatments, num_dosage_samples, \
                         model_folder,dataset_name ):

    ########
#    with tf.Session(graph=tf.Graph()) as sess:
#        print(model_folder)
#        tf.saved_model.loader.load(sess, ["serve"], model_folder)
#        rep1_pid1 = np.array([-1.9936499562340568,0.6104676956673598,0.6454972243679028,-0.19380063324460373,-0.18085983626508062,-0.5407380704358751,0.6947834398065859,-1.072908592877846,-0.31227235134089376,-0.453390318284153,-0.024221155956267643,-0.3335297064065907,-0.15862727729228462,0.3949605035632836,-0.011896637786808686,0.7144081947620927,1.024442845536566,-0.2952300592129878,0.18195428377590803,-0.8412131627350102,-0.5861635812080681,-0.6448792137659446,-0.7564450574430335,0.057790491077596665,-0.4886110933014305,1.5400813664319515,0.908780346631309,1.9011759323647124,1.5995626805123848,-0.3344360089421625,1.0751909674823816,0.9236285032698356,0.23322719306268336,-0.7280487609922841])
#        rep2_172 = np.array([-1.0251101183452827,-1.6380883167074154,0.6454972243679028,-0.19380063324460373,-0.18085983626508062,-0.5407380704358751,-0.5066473064622846,-1.5116001729444581,-2.5885364776068864,2.264362638853832,-0.5464529087694447,0.6489924321201418,-0.5200358162777989,-0.05496872987736581,-0.0943799931086835,0.09208178150855578,3.589297626148293,-0.34779115729521415,0.18195428377590803,1.4054643799595434,3.186908791034156,3.2741714983769903,-1.5117508904019687,-0.6483370717767875,-1.4085478413792965,1.5400813664319515,1.5811121708640317,1.6866900952680512,1.715233129984588,3.0251257172495616,1.5962127712223342,-0.8787723728682133,-2.078991558347544,0.5289729279084571])
#        rep3_6 = np.array([-0.6203767252389565,-1.6380883167074154,0.6454972243679028,-0.19380063324460373,-0.18085983626508062,-0.5407380704358751,-1.0816177350338154,-1.072908592877846,0.30852695582255946,-0.453390318284153,0.3674526586536152,-0.6680053280327123,-0.6405053292729702,0.2824781952031213,-0.7233155774379807,-0.46801199041962765,-0.20455007017322058,0.9399557457193316,-1.7660268719426442,-0.4533938250079741,-0.33462542305858634,-0.34757191836199747,-1.259982279415657,-0.7404406669317072,-0.696692024414281,0.26086797212219714,0.16174498637272844,0.5427656307525244,0.21151728684594656,-0.8143733983981232,2.1172345749622865,0.9236285032698356,-0.6579404507100084,0.5289729279084571])
#        dosages = np.random.rand(num_dosage_samples,1)
#        print('dosages',dosages.flatten().tolist())
#        treatment = np.ones((num_dosage_samples,1))
#        ZG = np.zeros((num_dosage_samples,num_dosage_samples))
#        for idx,features in enumerate([rep1_pid1, rep2_172, rep3_6]):
#            features  = np.repeat(np.expand_dims(features , axis=0),num_dosage_samples, axis=0)
#            batch_size = features.shape[0]
#            y = [np.ones((num_dosage_samples,1)),np.ones((num_dosage_samples,1)),np.zeros((num_dosage_samples,1))][idx]
#            real_d = [0.3621,0.3791,0.1278][idx] 
#            treatment_dosage_sample = np.ones((num_dosage_samples,1,num_dosage_samples)) * real_d
#            G_logits = sess.run('generator_outcomes:0',
#                        feed_dict={'input_features:0': features ,
#                                   'input_y:0': y,
#                                   'input_treatment:0':treatment,
#                                   'input_noise:0':ZG,
#                                   'input_dosage:0': dosages,
#                                   'input_treatment_dosage_samples:0': treatment_dosage_sample})
#
#            print(['rep1_pid1', 'rep2_172', 'rep3_6'][idx], G_logits.flatten().tolist())
#############




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

   ###
    patient = np.array([-0.0013521685087687012,-0.01553670947552708,-0.01579938616009668,-0.01553670947552708,-0.01579938616009668,-0.01579938616009668,-0.01579938616009668,0.01054708530223416,-0.012384589260691886,-0.015234631288272042,-0.00870711567671749,-0.01548417413861316,0.010205605612293682,-0.008851587853230772,-0.013514099004341164,0.023470778183058465,-0.014643608747990443,0.006291723012206646,-0.015379103464785321,-0.011228811848585648,0.02031865796822327,-0.008181762307578291,-0.00883845401900229,0.005214748605471289,0.009942928927724084,0.0015372750214968948,0.0039013651826232914,-0.010545852468704688,-0.006080348831021495,0.009942928927724084,262.660622506755])
    test_data1 = dict()
    test_data1['x'] = np.repeat(np.expand_dims(patient, axis=0), num_integration_samples, axis=0)
    test_data1['t'] = np.repeat(0, num_integration_samples)
    test_data1['d'] = treatment_strengths
    with tf.Session(graph=tf.Graph()) as sess:
         tf.saved_model.loader.load(sess, ["serve"], model_folder)
         pred_dose_response = get_model_predictions(sess=sess, num_treatments=num_treatments,
                                                           num_dosage_samples=num_dosage_samples, test_data=test_data1)
         pred_dose_response = pred_dose_response * (
                        dataset['metadata']['y_max'] - dataset['metadata']['y_min']) + \
                                     dataset['metadata']['y_min']
         print(pred_dose_response.tolist())
####
    
    with tf.Session(graph=tf.Graph()) as sess:
        print(model_folder)
        tf.saved_model.loader.load(sess, ["serve"], model_folder)
        #saver = tf.train.import_meta_graph('my_test_model-1000.meta')
        #saver.restore(sess,tf.train.latest_checkpoint('./'))

        for p_id, patient in enumerate(test_patients):
            for treatment_idx in range(num_treatments):
                test_data = dict()
                test_data['x'] = np.repeat(np.expand_dims(patient, axis=0), num_integration_samples, axis=0)
                test_data['t'] = np.repeat(treatment_idx, num_integration_samples)
                test_data['d'] = treatment_strengths

                pred_dose_response = get_model_predictions(sess=sess, num_treatments=num_treatments,
                                                           num_dosage_samples=num_dosage_samples, test_data=test_data)
                pred_dose_response = pred_dose_response * (
                        dataset['metadata']['y_max'] - dataset['metadata']['y_min']) + \
                                     dataset['metadata']['y_min']

                true_outcomes = response_data[test_ids[p_id]]

                mise = romb(np.square(true_outcomes - pred_dose_response), dx=step_size)
                mises.append(mise)

    return np.sqrt(np.mean(mises)), 0., 0. #np.sqrt(np.mean(dosage_policy_errors)), np.sqrt(np.mean(policy_errors))
#    return 0.5,0.,0.