#!/home/zoshs2/anaconda3/envs/tf_gpu/bin/python

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import argparse
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import StratifiedShuffleSplit
## Multi_gpu
# from keras.utils import multi_gpu_model



basepath = '/home/zoshs2/tf_gpu/Re/DATA'
Color = 'RGB'
Noise = 'WG' # WG : WHITE_GAUSSIAN_NOISE , UNI : UNIFORM_NOISE
Smoothing = 'BS2_BW1'

data = np.load("{}/{}_{}_{}_10318.npy".format(basepath, Smoothing, Noise, Color))

dtype = [('Img','i8',(200,200,3)),('xH','f8'),('Box','i8'),('z','f8'),('index','i8'),('xH_cat', 'i4')]
temp_data = np.empty(data.shape[0], dtype=dtype)
arr_cat = np.int32(np.ceil(data['xH'] / 0.01))
temp_data['Img'] = data['Img']
temp_data['xH'] = data['xH']
temp_data['Box'] = data['Box']
temp_data['z'] = data['z']
temp_data['index'] = data['index']
temp_data['xH_cat'] = arr_cat
del(data) # DON'T WASTE MEMORIES

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=21)
for train_index, test_index in split.split(temp_data, temp_data['xH_cat']):
    strat_train_set = temp_data[train_index]
    strat_test_set = temp_data[test_index]
    
del(temp_data) # DON'T WASTE MEMORIES
# Save the test dataset for inference after training.
np.save("{}_{}_{}_TEST_DATASET".format(Smoothing, Noise, Color), strat_test_set)
del(strat_test_set)
#del(testImagesX)
#del(testY)

'''
all_index = [ 3*i for i in range(67)]
np.random.shuffle(all_index)
test_index = all_index[:17]

indexes = []
for i in data['index']:
    indexes.append(i in test_index)
indexes = np.array(indexes)

test_data = data[indexes]
train_data = data[np.logical_not(indexes)]
del(data) # DON'T WASTE MEMORIES.

# np.random.seed(0)

np.random.shuffle(test_data)
'''

#trainImagesX = train_data['Img'] / 255.0
# testImagesX = test_data['Img'] / 255.0
#trainY = train_data['xH'] 
# testY = test_data['xH']
#print("Done read all dataset")


np.random.shuffle(strat_train_set)
np.random.shuffle(strat_train_set)

# Stratified K Fold Cross Validation for evaluation to my learning algorithm in last.
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
skfscores = []
n_sample = np.zeros(strat_train_set.shape[0])

opt = Adam(lr=1e-3, decay=1e-3/100)

for idx, (trainIDX, valIDX) in enumerate(skf.split(n_sample, n_sample)):
    fold_train_data = strat_train_set[trainIDX]
    fold_val_data = strat_train_set[valIDX]
    
    train_X = fold_train_data['Img'] / 255.0
    train_Y = fold_train_data['xH']
    val_X = fold_val_data['Img'] / 255.0
    val_Y = fold_val_data['xH']
    
    # Construct the learning algorithm
    
    ## CNN
    print("{}_Processing the training model".format(idx))
    model = Sequential()
    model.add(Conv2D(64, (10,10), padding='valid', input_shape=(200,200,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(128, (8,8), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(256, (5,5), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2))

    model.add(BatchNormalization())
    model.add(Conv2D(512, (3,3), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2))  
    model.add(Flatten())

    ## FC
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1024, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(512, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(256, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(128, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(64, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(32, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(10, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('linear'))
    model.add(Dense(1))
    
    ## Set-up the Checkpoint
    MODEL_SAVE_FOLDER_PATH = './{}_{}/{}th_CheckPointModels/'.format(Smoothing, Color, idx)
    if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
        os.makedirs(MODEL_SAVE_FOLDER_PATH, exist_ok=True)
        
    model_path = MODEL_SAVE_FOLDER_PATH + '{epoch:02d}-{val_loss:.4f}.hdf5'
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=2,
                                save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss', patience = 40)
    callbacks_list = [checkpoint, early_stopping]

    ## Compile the model & train
    model.compile(loss='MSE', optimizer=opt, metrics=['MSE'])
    hist = model.fit(train_X, train_Y, epochs=200, batch_size=16, validation_data=(val_X, val_Y), callbacks=callbacks_list)
    
    ## Model Evaluation 
    scores = model.evaluate(val_X, val_Y, verbose=2)
    skfscores.append(scores[1])
    
    ## Save the history plot for fitting model and Save the final trained model
    print("Being drawing the graph of MODEL LOSS")
    plt.clf()
    plt.plot(hist.history['loss'], 'r',lw=1.7, label='train_loss')
    plt.plot(hist.history['val_loss'], 'b', lw=1.7, label='val_loss')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE LOSS')
    plt.legend(['train','validation'], loc='upper right')
    
    plt.savefig("{}th_{}_{}_{}_MODEL_LOSS.pdf".format(idx, Smoothing, Noise, Color), bbox_inches="tight", dpi=150)
    plt.savefig("{}th_{}_{}_{}_MODEL_LOSS.png".format(idx, Smoothing, Noise, Color), bbox_inches="tight", dpi=150)
    
    model.save("{}th_{}_{}_{}_FinalModel_MSE.hdf5".format(idx, Smoothing, Noise, Color))
    
    ## Record the evaluation scores for each k-fold. 
    ### Could look whether my own model has been likely to be overfitted for ONLY train data.
    line = "{0}_th ::: LOSS = {1:0.5f}, MSE = {2:0.5f} \n".format(idx, scores[0], scores[1])
    with open("{}_{}_{}_SKFscores.txt".format(Smoothing, Noise, Color), 'a') as f:
        f.write(line)
    
skfscores_mean = np.mean(skfscores)
skfscores_std = np.std(skfscores)
line = "RESULT ::: K-fold Score :: MEAN = {0:0.5f}, STD = {1:0.5f}".format(skfscores_mean, skfscores_std)
with open("{}_{}_{}_SKFscores.txt".format(Smoothing, Noise, Color), 'a') as f:
    f.write(line)



    

