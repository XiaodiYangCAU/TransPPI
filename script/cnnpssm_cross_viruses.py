#!/usr/bin/env python
#-*-coding:utf-8

from __future__ import division
import time
from numpy.random import seed
seed(2066)
from tensorflow import set_random_seed
set_random_seed(2066)

start=time.ctime()
import os, sys
if '../embeddings' not in sys.path:
    sys.path.append('../embeddings')

from seq2pssm import s2p

import keras

from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout, merge
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D, GlobalMaxPooling1D
from keras.optimizers import Adam

import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score, average_precision_score

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

# Set GPU memory fraction
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.80)
session=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
KTF.set_session(session)


# Parameter setting
id1_index = 0
id2_index = 1
label_index = 2
if len(sys.argv) == 9:
    test_file, test_id2seq_file, train_virus, test_virus, batch_size, hidden_dim, dense_dim, n_epochs = sys.argv[1:]
    n_epochs = int(n_epochs)
else:
    print('please input all parameters!')


seq_size = 2000
# batch_size = 64
# hidden_dim = 64
# dense_dim = 512
# n_epochs = 100
lr = 0.0001
kernel_size = 3
pooling_size = 2

id2seqindex = {}
seqs = []
seqindex = 0
print(test_id2seq_file)
for line in open(test_id2seq_file):
    line = line.strip().split('\t')
    id2seqindex[line[0]] = seqindex
    seqs.append(line[1])
    seqindex += 1


from tqdm import tqdm
seq2p = s2p()
index2id1,index2id2={},{}
raw_data = []
id_array = []
seq_array = []
id2index = {}
index = 0
for line in tqdm(open(test_file)):
    line = line.strip().split('\t')
    if line[id1_index] not in id2index:
        id2index[line[id1_index]] = index
        index += 1
        id_array.append(line[id1_index])
        seq_array.append(seqs[id2seqindex[line[id1_index]]])
    id1=line[id1_index]
    line[id1_index] = id2index[line[id1_index]]
    index1=line[id1_index]
    index2id1[index1]=id1
    if line[id2_index] not in id2index:
        id2index[line[id2_index]] = index
        index += 1
        id_array.append(line[id2_index])
        seq_array.append(seqs[id2seqindex[line[id2_index]]])
    id2=line[id2_index]
    line[id2_index] = id2index[line[id2_index]]
    index2=line[id2_index]
    index2id2[index2]=id2
    raw_data.append(line)
print(len(raw_data))

dim = 20
seq_tensor = np.array([seq2p.pssm_normalized(id, line, seq_size) for id, line in zip(tqdm(id_array),tqdm(seq_array))], dtype='int32')
del id_array
del seq_array

seq_index1 = np.array([line[id1_index] for line in tqdm(raw_data)])
seq_index2 = np.array([line[id2_index] for line in tqdm(raw_data)])


class_map = {'0':1,'1':0}
class_labels = np.zeros((len(raw_data), 2))
for i in range(len(raw_data)):
    class_labels[i][class_map[raw_data[i][label_index]]] = 1.
print(class_labels)


from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit
kf = StratifiedKFold(n_splits=5, random_state=2066, shuffle=True)
train_test = []
print(class_labels[:,0])
for train, test in kf.split(class_labels[:,0],class_labels[:,0]):
    print(np.sum(class_labels[train], 0)[0],np.sum(class_labels[train], 0)[1])
    train_test.append((train, test))

print (len(train_test))

# initialization
n_model = 0
n_hit = 0
n_total = 0
n_pos = 0
n_true_pos = 0
n_false_pos = 0
n_true_neg = 0
n_false_neg = 0

from keras.utils import plot_model
from collections import Counter
from keras.models import load_model

train_file='../results/'+train_virus+'_cnnpssm.txt'
for train, test in train_test:
    n_model+=1
    # Load model of train dataset
    merge_model=load_model(train_file+str(n_model)+'.h5')
    print(seq_index1[train][0])
    print(merge_model.get_weights()[0][0][0])
    print(class_labels[train][:,0])
    # Set class weight for samples
    counter = Counter(class_labels[train][:,0])
    majority = max(counter.values())
    class_weight = {cls: float(majority / count) for cls, count in counter.items()}
    print(class_weight)
    # Test the model
    pred = merge_model.predict([seq_tensor[seq_index1[test]], seq_tensor[seq_index2[test]]])
    # Output prediction scores (Format: 'label\tscore_t\tscore_f\tid1\tid2\n')
    w = open(test_virus+'_'+train_virus+'_cross_viral_test_score'+str(n_model),'w')
    for i in range(len(class_labels[test])):
        n_total += 1
        w.write(str(class_labels[test][i][0])+'\t'+str(pred[i][0])+'\t'+str(pred[i][1])+'\t'+str(index2id1[seq_index1[test][i]])+'\t'+str(index2id2[seq_index2[test][i]])+'\n')
        if np.argmax(class_labels[test][i]) == np.argmax(pred[i]):
            n_hit += 1
        if class_labels[test][i][0] > 0.:
            n_pos += 1.
            if pred[i][0] > pred[i][1]:
                n_true_pos += 1
            else:
                n_false_neg += 1
        else:
            if pred[i][0] > pred[i][1]:
                n_false_pos += 1
            else:
                n_true_neg += 1
    w.close()

# Calculate metrics
accuracy = n_hit / n_total
prec = n_true_pos / (n_true_pos + n_false_pos)
recall = n_true_pos / n_pos
spec = n_true_neg / (n_true_neg + n_false_pos)
f1 = 2. * prec * recall / (prec + recall)
print (accuracy, prec, recall, f1)


result_file = '../results/'+test_virus+'_'+train_virus+'_cross_viral_test.txt'
basename = test_virus+'_'+train_virus+'_cross_viral_test_score'
os.system('cat '+basename+'1 '+basename+'2 '+basename+'3 '+basename+'4 '+basename+'5 > '+ result_file)
data=np.genfromtxt(result_file, dtype=str)
y = data[:,0]
x = data[:,1]
y = y.astype(float)
x = x.astype(float)
auc = roc_auc_score(y,x)
auprc = average_precision_score(y,x)

end=time.ctime()
w = open('../Run_result.txt','a')
if os.popen("grep $'Source' ../Run_result.txt").read():pass
else:w.write('Source\tTarget\tMethod\tBatch_size\tn_epochs\tlearning_rate\tAUC\tAUPRC\tAccuracy\tPrecision\tRecall\tSpecificity\tF1\tStart\tEnd\n')
w.write('Human-' + train_virus + '\tHuman-' + test_virus + '\tCross viral test\t' + str(batch_size) + '\t' + str(seq_size) + '\t' + str(n_epochs) + '\t' + str(lr) + '\t%.3f'%auc + '\t%.3f'%auprc + '\t%.3f'%accuracy + '\t%.3f'%prec + '\t%.3f'%recall + '\t%.3f'%spec + '\t%.3f'%f1 + '\t'+str(start) + '\t' + str(end) + '\n')











