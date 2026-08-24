## E2LC
This repo hosts the implementation of our proposed method E2LC. To run our codes first assure MIMIC-III and MIMIC-IV are installed as a database, which can be queried using SQL tools. Python requires version = 3.7 or newer. To install the dependencies, try the following code in command line:
```console
pip install -r requirements.txt
```
Another prerequisite for ajusting the level of bias is to get the propensity scores pre-calculated. This requires to run the file "cal_propensity.py" under "/VCNet_E2LC/".

### Data Preparation
We provide synthetic data used for evaluation. For the preparation of data using real-world datasets, please refer to [ADMIT](https://papers.nips.cc/paper_files/paper/2022/hash/390bb66a088d37f62ee9fb779c5953c2-Abstract-Conference.html). 

### Training
This repo contains implementation for VCNet-E2LC and TransTEE-E2LC. First pretrain the plug-in estimator by running code for the associated model. To pretarin VCNet, go to "/VCNet/" and run "main.py". The pretrained plug-in esitmator will be saved automatically under "/VCNet-E2LC/". Then, run "run_data_aug.py" under "/VCNet-E2LC/" to implement VCNet-E2LC. Note that the configurations are fixed so that the model configurations should be consistent in both files. It is also similar for TransTEE. The file names "bias" are for ajusting the level of bias. In summary, for bias level = 0, the plug-in estimator can be trained by running "/VCNet/main.py" or "/TransTEE/test_TransTEE.py". Then, run "/VCNet_E2LC/run_data_aug.py" or "/TransTEE_E2LC/test_TransTEE.py" to apply E2LC. For bias level > 0, the plug-in estimator can be trained by running "/VCNet/run_data_aug_bias.py" or "/TransTEE/test_TransTEE_bias.py". Then, run "/VCNet_E2LC/run_data_aug_bias.py" or "/TransTEE_E2LC/test_TransTEE_bias.py" to apply E2LC.
