## E2LC
This repo hosts the implementation of our proposed method E2LC. To run our codes first assure MIMIC-III and MIMIC-IV are installed as a database, which can be queried using SQL tools. Python requires version = 3.7 or newer. To install the dependencies, try the following code in command line:
```console
pip install -r requirements.txt
```
Another prerequisite for ajusting the level of bias is to get the propensity scores pre-calculated. This requires to run the file "cal_propensity.py" under "/VCNet_E2LC/".

### Data Preparation
To generate synthetic data, go to "/data" directory and run "generate_synthetic.py". For the preparation of data using real-world datasets, please refer to [ADMIT](https://papers.nips.cc/paper_files/paper/2022/hash/390bb66a088d37f62ee9fb779c5953c2-Abstract-Conference.html). 

### Training
The training of E2LC consists of stage. First, pretrain plug-in. Second, train E2LC. Take VCNet as an example,  go to "/VCNet/" and run "main.py" for the first step. Then, change the hyperparameters in "run_data_aug.py" according to those in the plug-in and run "run_data_aug.py" under "/VCNet-E2LC/" to implement VCNet-E2LC.
