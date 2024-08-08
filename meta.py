import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers, utils, models, Sequential
from tensorflow.keras import losses, Input
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D

# from config import args
import shutil
import os

from dataloader import *


# def get_maml_model(input_shape=(5971,1),num_classes=4):
#     model = Sequential()
#     model.add(Input(shape=input_shape))
#     # model.add(layers.Dense(128,activation="tanh"))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(128,activation="tanh",kernel_regularizer='l2'))
#     model.add(layers.Dropout(0.6))
#     model.add(layers.Dense(64,activation="tanh",kernel_regularizer='l2'))
#     # model.add(layers.Dense(128,activation="tanh"))
#     model.add(layers.Dropout(0.6))
#     model.add(layers.Dense(num_classes,activation="tanh"))

#     return model

def get_maml_model(input_shape=(None,1),num_classes=4,conv_layer=4,model_type='lecun'):
        """
        return: MAML model
        """
        if model_type == 'orign':
            model = Sequential()
            model.add(Input(shape=input_shape))

            for i in range(conv_layer):
                model.add(Conv1D(filters=64, 
                                kernel_size=3, 
                                padding='same', 
                                activation="relu",
                                name="C%02d"%i))
                model.add(BatchNormalization())
                model.add(MaxPooling1D(pool_size=2, strides=2))

            model.add(Flatten())
            model.add(layers.Dense(64,activation='relu')),
            model.add(layers.Dense(64,activation='relu')),
            model.add(layers.Dense(num_classes, activation='sigmoid'))

        if model_type == "lecun":
            model = Sequential([
            Conv1D(filters=6, kernel_size=21, strides=1, padding='same', activation='relu', input_shape= input_shape,
                    kernel_initializer=keras.initializers.he_normal()),
            BatchNormalization(),
            MaxPooling1D(pool_size=2, strides=2, padding='same'),
            Conv1D(filters=16, kernel_size=5, strides=1, padding='same',activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2, strides=2, padding='same'),
            Flatten(),
            Dense(120, activation='relu'),
            Dense(84),
            Dense(num_classes, activation='sigmoid') # or Activation('softmax')
            ])
        
        if model_type == "lenet":
            model = Sequential([
            Conv1D(filters=16, kernel_size=21, strides=1, padding='same', input_shape= input_shape,
                kernel_initializer=keras.initializers.he_normal(), activation='relu'),
            BatchNormalization(),
            # LeakyReLU(),
            MaxPooling1D(pool_size=2, strides=2, padding='same'),
            Conv1D(filters=32, kernel_size=11, strides=1, padding='same', activation='relu'),
            BatchNormalization(),
            # LeakyReLU(),
            MaxPooling1D(pool_size=2, strides=2, padding='same'),
            Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu'),
            BatchNormalization(),
            # LeakyReLU(),
            MaxPooling1D(pool_size=2, strides=2, padding='same', activation='relu'),
            Flatten(),
            Dense(2050, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='sigmoid') # or Activation('softmax')
        ])

        if model_type == 'vgg':
            model = Sequential([
            Conv1D(filters=64, kernel_size=21, strides=1, padding='same', activation='relu',
                   input_shape= input_shape, kernel_initializer=keras.initializers.he_normal()),
            BatchNormalization(),
            Conv1D(filters=64, kernel_size=21, strides=1, padding='same',activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2, strides=2, padding='same'),
            Conv1D(filters=128, kernel_size=11, strides=1, padding='same',activation='relu'),
            BatchNormalization(),
            Conv1D(filters=128, kernel_size=11, strides=1, padding='same',activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2, strides=2, padding='same'),
            Conv1D(filters=256, kernel_size=5, strides=1, padding='same',activation='relu'),
            BatchNormalization(),
            Conv1D(filters=256, kernel_size=5, strides=1, padding='same',activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2, strides=2, padding='same'),
            Flatten(),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='sigmoid') # or Activation('softmax')
        ])
        return model

# train_list = [['IGR37','IGR39'],['WM115','WM2664']]
def train_on_batch(train_list, in_optimizer, outer_optimizer=None,
                   input_shape=(1354,1),num_classes=1, 
                   inner_step=1,k_shot=1, meta_batch_size=10):

    data_train, _, labels_train, _ = get_dataset(train_list=train_list)
    meta_support_data, meta_support_labels, meta_query_data, meta_query_labels = get_one_batch(data_train,labels_train,
                                                                                               k_shot=k_shot,
                                                                                               meta_batch_size=meta_batch_size)
    
    #load model
    meta_model = get_maml_model(input_shape=input_shape,num_classes=num_classes)
    # print(meta_model.summary())
    meta_weights = meta_model.get_weights()

    batch_acc = []
    batch_loss = []
    task_weights = []

    for support_data, support_labels in zip(meta_support_data, meta_support_labels):
        meta_model.set_weights(meta_weights)
        for step in range(inner_step):
            # print("step %02d\n"%step)
            with tf.GradientTape() as tape:
                # print("support labels",support_labels)
                logits = meta_model(support_data, training=True)
                # print("inner logits",logits)
                bce = losses.BinaryCrossentropy()
                support_labels_2 = np.expand_dims(support_labels, axis=1)
                loss = bce(support_labels_2, logits)
                loss = tf.reduce_mean(loss)
                
                acc = np.mean(support_labels==logits)
            grads = tape.gradient(loss, meta_model.trainable_variables)
            in_optimizer.apply_gradients(zip(grads, meta_model.trainable_variables))

        
        task_weights.append(meta_model.get_weights())

    with tf.GradientTape() as tape:
        for i, (query_data, query_labels) in enumerate(zip(meta_query_data, meta_query_labels)):
                
            meta_model.set_weights(task_weights[i])
            logits = meta_model(query_data, training=True)
            query_labels = np.expand_dims(query_labels, axis=1)
            loss = bce(query_labels, logits)
            loss = tf.reduce_mean(loss)
            batch_loss.append(loss)
            
            acc = np.mean(support_labels ==logits)
            batch_acc.append(acc)

        mean_acc = tf.reduce_mean(batch_acc)
        mean_loss = tf.reduce_mean(batch_loss)

    meta_model.set_weights(meta_weights)
        
    if outer_optimizer:
        grads = tape.gradient(mean_loss, meta_model.trainable_variables)
        outer_optimizer.apply_gradients(zip(grads, meta_model.trainable_variables))
    
    return mean_loss, mean_acc