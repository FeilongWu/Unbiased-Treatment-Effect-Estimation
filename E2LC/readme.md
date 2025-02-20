## E2LC
This repo hosts the implementation of our proposed method E2LC. To run our codes first assure MIMIC-III and MIMIC-IV are installed as a database, which can be queried using SQL tools. Python requires version = 3.7 or newer. To install the dependencies, try the following code in command line:
```console
pip install -r requirements.txt
```
Another prerequisite for ajusting the level of bias is to get the propensity scores pre-calculated. This requires to run the file "cal_propensity.py" under "/VCNet_E2LC/".

### Data Preparation
Go to directory "data_preparation", there are five folders named by the corresponding database. To prepare the specific dataset, just click the go to the associated sub-folder. Then, run the Jupyter Notebook to extract the raw data files, which will be saved automatically. Next, run the "extract_data.py" file to filter the data. This will output a CSV file for training. Finally, run "calibrate_response_IPW.py" to prepare synthetic reference curves for evaluation. This will output a pickle file for testing. For XXX (e.g., MIMICIII-Seda) dataset, place the two files, namely, "XXX.csv" and "XXX_response_curve_calibrate.pickle" under the directory "./E2LC/data/".

### Training
This repo contains implementation for VCNet-E2LC and TransTEE-E2LC. First pretrain the plug-in estimator by running code for the associated model. To pretarin VCNet, go to "/VCNet/" and run "main.py". The pretrained plug-in esitmator will be saved automatically under "/VCNet-E2LC/". Then, run "run_data_aug.py" under "/VCNet-E2LC/" to implement VCNet-E2LC. Note that the configurations are fixed so that the model configurations should be consistent in both files. It is also similar for TransTEE. The file names "bias" are for ajusting the level of bias. In summary, for bias level = 0, the plug-in estimator can be trained by running "/VCNet/main.py" or "/TransTEE/test_TransTEE.py". Then, run "/VCNet_E2LC/run_data_aug.py" or "/TransTEE_E2LC/test_TransTEE.py" to apply E2LC. For bias level > 0, the plug-in estimator can be trained by running "/VCNet/run_data_aug_bias.py" or "/TransTEE/test_TransTEE_bias.py". Then, run "/VCNet_E2LC/run_data_aug_bias.py" or "/TransTEE_E2LC/test_TransTEE_bias.py" to apply E2LC.
