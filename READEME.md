### TransPPI ###
TransPPI propose two types of deep transfer learning methods based on convolutional neural network (CNN) layers with pairwise position specific scoring matrix (PSSM) feature inputs to predict target human-virus protein-protein interactions (PPIs).

### Usage ###

# Step 0 Prepare PSSM files
  cat ./x* > pssm.tar.gz
  tar -zxvf pssm.tar.gz

# Step 1 Baseline method: pre-train DNN models (PSSM+CNN+MLP) on human-virus PPI source dataset.
  ./1_run_dnn.sh

# Step 2 Two types of transfer learning methods and cross-viral tests.
a. Import and frozen the parameters of CNN layers of source domain, then train the DNN model (PSSM+CNN+MLP) on target dataset.
  ./2_run_transfer_frozen.sh

b. Import the parameters of CNN layers of source domain, then fine-tune parameters on target dataset.
  ./2_run_transfer_finetune.sh

c. Train the DNN model (PSSM+CNN+MLP) on a human-virus system and test the model on another human-virus system.
  ./2_run_cross_viruses.sh

### Output ###
a. Run_result.txt - performance of various methods
b. xx.h5 - deep learning models
c. xx.txt - prediction result files containing label, pairwise protein ids and prediction score.


### Notice ###
The human-virus systems and related parameters can be changed in the above shell scripts.

### Dataset ###
  Eight human-virus PPI systems. The ratio of positive-to-negative is 1:10.
  PSSM of human and viral proteins are deposited in 'script/pssm/'.
  9880 human-HIV PPIs, 98800 non-human-HIV PPIs.
  5966 human-Herpes PPIs, 59660 non-human-Herpes PPIs.
  5099 human-Papilloma PPIs, 50990 non-human-Papilloma PPIs.
  3044 human-Influenza PPIs, 30440 non-human-Influenza PPIs.
  1300 human-Hepatitis PPIs, 13000 non-human-Hepatitis PPIs.
  927 human-DENV PPIs, 9270 non-human-DENV PPIs.
  709 human-ZIKV PPIs, 7090 non-human-ZIKV PPIs.
  568 human-SARS-CoV-2 PPIs, 5680 non-human-SARS-CoV-2 PPIs.
  
### Requirements ###
  - Tensorflow (==1.7.0)
  - Keras (==2.2.4)
  - scikit-learn (==0.22.1)
  - numpy (==1.16.6)

###Use the following command to install all dependent packages ###
  pip install requirements.txt

### Citation ###
Please kindly cite the paper if you use refers to the paper, code or datasets.
@article{Yang2021Transfer,
  title={Transfer learning via multi-scale convolutional neural layers for human-virus PPI prediction},
  author={Yang, Xiaodi and Yang, Shiping and Lian, Xianyi and Wuchty, Stefan and Zhang, Ziding},
  journal={xx}
}


