cd script
mkdir ../results
#step2: Import the parameters of CNN layers of source domain, then fine-tune parameters on target dataset. (5-fold cross-validation)
# python script.py <ppi_target_dataset> <sequence_file_of_target_dataset> <source_virus> <target_virus> <batch_size> <hidden_dim> <dense_dim> <epochs>
CUDA_VISIBLE_DEVICES=1 python cnnpssm_transfer_finetune.py ../sample/SARS2/protein_pair_label.txt ../sample/SARS2/pro_seq.txt Herpes SARS2 64 64 512 100
