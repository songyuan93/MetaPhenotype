import sys
import argparse
import pickle
import pandas as pd
# from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers, utils

from meta import *
from dataloader import *

train_list = [['IGR37','IGR39'],['WM115','WM2664']]
in_optimizer = optimizers.legacy.Adam(learning_rate=0.01)
outer_optimizer = optimizers.legacy.Adam(learning_rate=0.01)

k_shot = 5
meta_batch_size = 50
inner_step = 5

for epoch in range(50):
    print('\nEpoch {}/50'.format(epoch+1))

    # train_progbar = utils.Progbar(train_data.steps)
    # val_progbar = utils.Progbar(val_data.steps)

    train_meta_loss = []
    train_meta_acc = []
    val_meta_loss = []
    val_meta_acc = []
    
    steps = 5
    progbar = utils.Progbar(steps)

    for i in range(steps):
        batch_train_loss, acc = train_on_batch(train_list=train_list,
                                               in_optimizer=in_optimizer,
                                               inner_step=inner_step,
                                               outer_optimizer=outer_optimizer,
                                               k_shot= k_shot,
                                               meta_batch_size=meta_batch_size)

        train_meta_loss.append(batch_train_loss)
        train_meta_acc.append(acc)
        # print("loss: %.3f"%np.mean(train_meta_loss))
        # print("accuracy: %.3f"%np.mean(train_meta_acc))
        progbar.update(i+1, [('loss', np.mean(train_meta_loss)),
                             ('accuracy', np.mean(train_meta_acc))])

    # batch_val_loss, val_acc = train_on_batch(get_one_batch(), 
    #                                          inner_optimizer, 
    #                                          inner_step=3)

    # val_meta_loss.append(batch_val_loss)
    # val_meta_acc.append(val_acc)

    # model.save_weights("maml.h5")

results = {}
# results['predict_testing_eval'] = model.evaluate(x_test,y_test)
results['loss'] = train_meta_loss
results['accuracy'] = train_meta_acc
fp = open("results/meta_step%02d_size%03d_shot%02d.pkl"%(inner_step,meta_batch_size,k_shot), "wb")
pickle.dump(results, fp)
fp.close()