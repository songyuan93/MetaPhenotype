import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas
import tensorflow_datasets as tfds
import glob
# import seaborn

learning_rate = 0.003
meta_step_size = 0.25

inner_batch_size = 25
eval_batch_size = 25

meta_iters = 2000
eval_iters = 5
inner_iters = 4

eval_interval = 1
train_shots = 20

shots = 1
classes = 7

def get_mz(df):
    cut = 0
    for i, mz in enumerate(df.iloc[:,0]):
        if mz < 1000:
            cut = i
    return cut

def get_data(df):
    '''
    return all cells labels and mass spec m/z data
    '''
    col_label = df.iloc[[0]]
    labels = []
    all_data = []

    for col_num, col in enumerate(col_label):
        if col_num > 0:
            #get label
            labels.append(col_label[col].values[0])
            #get m/z
            mz = df.iloc[1:,[col_num]]
            mz = mz.to_numpy().reshape(-1).astype(np.float64)
            all_data.append(mz)
    
    labels = np.array(labels)
    all_data = np.tanh(all_data)

    return  all_data, labels

def rm_zero(data, labels, train_list):

    mask = [x in train_list for x in labels]
    labels = labels[mask]
    data = data[mask]
    labels[labels==train_list[0]] = 0.
    labels[labels==train_list[1]] = 1.
    labels = labels.astype(np.float64)
    # print("data: ",data.shape)
    # print("threshold: ",np.ceil(data.shape[0]/2))
    mask = data == 0
    data_del = np.argwhere(np.count_nonzero(mask,axis=0) > np.ceil(data.shape[0]/2))
    data = np.delete(data,data_del,axis=1)
    # print("labels",labels)

    return data, labels

def unify_shape(train_data,test_data):
    train_shape = train_data.shape
    test_shape = test_data.shape

    if train_shape[1] >= test_shape[1]:
        delta = train_shape[1] - test_shape[1]
        concat = np.zeros((test_data.shape[0],delta))
        test_data = np.concatenate((test_data,concat),axis=1)
    
    else:
        delta = -(train_shape[1] - test_shape[1])
        concat = np.zeros((train_data.shape[0],delta))
        train_data = np.concatenate((train_data,concat),axis=1)
    
    return train_data, test_data

def get_dataset(train_list,filename="dataset/dataset.csv"):
    
    df = pandas.read_csv(filename,low_memory=False)
    data, labels = get_data(df)
    data_train, labels_train = rm_zero(data=data,labels=labels,train_list=train_list[0])
    data_test, labels_test = rm_zero(data=data,labels=labels,train_list=train_list[1])
    data_train, data_test = unify_shape(data_train,data_test)

    return data_train, data_test, labels_train, labels_test

def get_one_task(data,labels,k_shot=1):
    
    support_data = []
    query_data = []

    support_label = []
    query_label = []

    for i in range(k_shot):
        support_1 = np.random.randint(0,int(labels.sum()))
        support_0 = np.random.randint(int(labels.sum()), data.shape[0])
        query_num = np.random.randint(0, data.shape[0])
        

        select = np.random.randint(0,2)
        # print(select)

        if select == 0:
            support_data.append(data[support_0])
            support_data.append(data[support_1])
            support_label.append(labels[support_0])
            support_label.append(labels[support_1])
        
        elif select == 1:
            support_data.append(data[support_1])
            support_data.append(data[support_0])
            support_label.append(labels[support_1])
            support_label.append(labels[support_0])

        query_data.append(data[query_num])
        query_label.append(labels[query_num])
    # print(support_label,query_label)
    return np.array(support_data), np.array(support_label),\
           np.array(query_data), np.array(query_label)

# support_data, support_label, query_data, query_label = get_one_task(data_igr,labels_igr,k_shot=1)
# print(support_data.shape)
# print(support_label.shape)
# print(query_data.shape)
# print(query_label.shape)

def get_one_batch(data,labels,k_shot=1,meta_batch_size = 10):
    
    batch_support_data = []
    batch_support_label = []
    batch_query_data = []
    batch_query_label = []

    for _ in range(meta_batch_size):
        support_data, support_label, query_data, query_label = get_one_task(data,labels,k_shot=k_shot)
        batch_support_data.append(support_data)
        batch_support_label.append(support_label)
        batch_query_data.append(query_data)
        batch_query_label.append(query_label)

    batch_support_data = np.array(batch_support_data)
    batch_support_label= np.array(batch_support_label)
    batch_query_data = np.array(batch_query_data)
    batch_query_label = np.array(batch_query_label)

    # print("batch query spec: ",batch_query_spec.shape)
    return batch_support_data,batch_support_label,batch_query_data,batch_query_label


# filename = "dataset/dataset.csv"
# # cell_list = [['IGR37','IGR39'],
# #              ['IGR37_treated','IGR39_treated'],
# #              ['WM115','WM2664'],
# #              ['WM115_treated','WM2664_treated']]

# train_list = [['IGR37','IGR39'],
#               ['WM115','WM2664']]

# data_igr, data_wm, labels_igr, labels_wm = get_dataset(train_list=train_list)
# support_data, support_label, query_data, query_label = get_one_batch(data_igr,labels_igr,k_shot=1)

# print(support_data.shape)
# print(support_label.shape)
# print(query_data.shape)
# print(query_label.shape)