#!/usr/bin/env python
#-*-coding:utf-8

### Reference ###
# Author information: Xiaodi Yang, China Agricultural University, email: xiao_di_yang@163.com.
# Cititation: Yamg,X. et al. (2021) Transfer learning via multi-scale convolutional neural layers for human-virus protein-protein interaction prediction. Bioinformatics.
# Part of the code was modified from https://github.com/muhaochen/seq_ppi and the corresponding reference is: 
# Chen,M. et al. (2019) Multifaceted protein-protein interaction prediction based on Siamese residual RCNN. Bioinformatics, 35, i305–i314.

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
from numpy import linalg as LA
import scipy
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
print(sys.argv)

target_file, id2seq_file, result_file, virus, batch_size, hidden_dim, dense_dim, n_epochs = sys.argv[1:]
batch_size = int(batch_size)
hidden_dim = int(hidden_dim)
dense_dim = int(dense_dim)
n_epochs = int(n_epochs)

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
print(id2seq_file)
for line in open(id2seq_file):
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
for line in tqdm(open(target_file)):
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
print (len(raw_data))

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

def build_model():
    # seq_size=2000, dim=20
    seq_input1 = Input(shape=(seq_size, dim), name='seq1')
    seq_input2 = Input(shape=(seq_size, dim), name='seq2')
    l1=Conv1D(hidden_dim, kernel_size, name='pro_conv1')  
    l2=Conv1D(hidden_dim*2, kernel_size, name='pro_conv2')
    l3=Conv1D(hidden_dim*2*2, kernel_size, name='pro_conv3')
    l4=Conv1D(hidden_dim*2*2*2, kernel_size, name='pro_conv4')
    v1=GlobalMaxPooling1D(name='pro1_globalmax_pooling')(l4(MaxPooling1D(pooling_size, name='pro1_maxpooling3')(l3(MaxPooling1D(pooling_size, name='pro1_maxpooling2')(l2(MaxPooling1D(pooling_size, name='pro1_maxpooling1')(l1(seq_input1))))))))
    v2=GlobalMaxPooling1D(name='pro2_globalmax_pooling')(l4(MaxPooling1D(pooling_size, name='pro2_maxpooling3')(l3(MaxPooling1D(pooling_size, name='pro2_maxpooling2')(l2(MaxPooling1D(pooling_size, name='pro2_maxpooling1')(l1(seq_input2))))))))
    merge_vector = concatenate([v1, v2], name='merge')
    x = Dense(dense_dim, activation='relu', name='Dense_1')(merge_vector)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    x = Dense(int((dense_dim)/2), activation='relu', name='Dense_2')(x)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    main_output = Dense(2, activation='softmax', name='Dense_3')(x)
    merge_model = Model(inputs=[seq_input1, seq_input2], outputs=[main_output])
    return merge_model, merge_vector

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

for train, test in train_test:
    n_model += 1
    merge_model = None
    merge_model, merge_vector = build_model()
    adam = Adam(lr=lr, amsgrad=True, epsilon=1e-6)
    # Compile model
    merge_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # Plot model architecture
    plot_model(merge_model, to_file='../results/model.png',show_shapes='True')
    print(len(test))
    # Set class weight for samples
    counter = Counter(class_labels[train][:,0])
    majority = max(counter.values())
    class_weight = {cls: float(majority / count) for cls, count in counter.items()}
    print(class_weight)
    # Train model
    merge_model.fit([seq_tensor[seq_index1[train]], seq_tensor[seq_index2[train]]], class_labels[train], batch_size=batch_size, epochs=n_epochs, class_weight=class_weight)
    # Save model
    merge_model.save(result_file+str(n_model)+'.h5')
    pred = merge_model.predict([seq_tensor[seq_index1[test]], seq_tensor[seq_index2[test]]])
    # Output prediction scores (Format: 'label\tscore_t\tscore_f\tid1\tid2\n')
    w = open(result_file+'score'+str(n_model),'w')
    for i in range(len(class_labels[test])):
        n_total += 1
        w.write(str(class_labels[test][i][0])+'\t'+str(pred[i][0])+'\t'+str(pred[i][1])+'\t'+str(index2id1[seq_index1[test][i]])+'\t'+str(index2id2[seq_index2[test][i]])+'\n')
        if np.argmax(class_labels[test][i]) == np.argmax(pred[i]):
            n_hit += 1
        if class_labels[test][i][0] > 0:
            n_pos += 1
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

basename = result_file+'score'
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
else:w.write('Source\tTarget\tMethod\tBatch_size\tSequence_size\tn_epochs\tlearning_rate\tAUC\tAUPRC\tAccuracy\tPrecision\tRecall\tSpecificity\tF1\tStart\tEnd\n')
w.write('Human-' + virus + '\tHuman-' + virus + '\tDNN\t' + str(batch_size) + '\t' + str(seq_size) + '\t' + str(n_epochs) + '\t' + str(lr) + '\t%.3f'%auc + '\t%.3f'%auprc + '\t%.3f'%accuracy + '\t%.3f'%prec + '\t%.3f'%recall + '\t%.3f'%spec + '\t%.3f'%f1 + '\t'+str(start) + '\t' + str(end) + '\n')
