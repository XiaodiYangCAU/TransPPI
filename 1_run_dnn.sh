cd script
mkdir ../results
# step1: Pre-train DNN model (PSSM+CNN+MLP) on human-virus PPI source dataset. (5-fold cross-validation)
# python script.py <ppi_dataset> <sequence_file>  <result_file> <virus_abbr> <batch_size> <hidden_dim> <dense_dim> <n_epochs>
CUDA_VISIBLE_DEVICES=0 python cnn_pssm.py ../sample/SARS2/protein_pair_label.txt ../sample/SARS2/pro_seq.txt ../results/SARS2_cnnpssm.txt SARS2 64 64 512 100
CUDA_VISIBLE_DEVICES=0 python cnn_pssm.py ../sample/Herpes/protein_pair_label.txt ../sample/Herpes/pro_seq.txt ../results/Herpes_cnnpssm.txt Herpes 64 64 512 100
