cd script
mkdir ../results
#step2: Train the DNN model (PSSM+CNN+MLP) on a human-virus system and test the model on another human-virus system. (5-fold cross-validation)
# python script.py <ppi_test_dataset> <sequence_file_of_test_dataset> <virus_of_train_dataset> <virus_of_test_dataset> <batch_size> <hidden_dim> <dense_dim> <epochs>
CUDA_VISIBLE_DEVICES=1 python cnnpssm_cross_viruses.py ../sample/SARS2/protein_pair_label.txt ../sample/SARS2/pro_seq.txt Herpes SARS2 64 64 512 100
