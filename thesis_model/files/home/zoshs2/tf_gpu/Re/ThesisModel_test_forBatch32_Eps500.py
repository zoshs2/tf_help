#!/home/zoshs2/anaconda3/envs/tf_gpu/bin/python
# Simplified_model.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint # , EarlyStopping, TensorBoard
import argparse
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import StratifiedShuffleSplit
import time


epochs = 500 
batch_size = 32 # ['4', 8, 16, 32, 64, 128, 256] 

# Target Configuration
basepath = '/home/zoshs2/tf_gpu/Re/DATA'
Color = 'RGB' # To date, this is only existed. - RGB
Noise = 'NN' # WG : WHITE_GAUSSIAN_NOISE , UNI : UNIFORM_NOISE, NN : Without Nowise
Smoothing = 'BS2_BW1' # BS{a}_BW{b} where a, b are BEAMSIZE and BANDWIDTH that we had convolved, respectively. 

# Save Configuration
TARGET_FOLDER_PATH = './{}_{}_Batch{}_Eps{}/'.format(Smoothing, Noise, batch_size, epochs)
if not os.path.exists(TARGET_FOLDER_PATH):
    os.makedirs(TARGET_FOLDER_PATH, exist_ok=True)

# Targeting Whole Dataset Load
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
np.save(TARGET_FOLDER_PATH+"{}_{}_Batch{}_Eps{}_TEST_DATASET".format(Smoothing, Noise, batch_size, epochs), strat_test_set)
del(strat_test_set)

#trainImagesX = train_data['Img'] / 255.0
# testImagesX = test_data['Img'] / 255.0
#trainY = train_data['xH'] 
# testY = test_data['xH']
#print("Done read all dataset")

# To compare with performances of basic setups which are the Batch_size, and Epochs.
np.random.seed(0)
np.random.shuffle(strat_train_set)
np.random.shuffle(strat_train_set)

# Stratified K Fold Cross Validation for evaluation to my learning algorithm in last.
n_splits = 10
random_state = 5
stp=1 # Stopping point for K-fold CV
lr = 1e-3 # 상당히 좋은 1e-3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
skfscores = []
n_sample = np.zeros(strat_train_set.shape[0])
opt = Adam(lr=lr) # , decay=1e-3/100) 
Initial_Time = time.time()

for idx, (trainIDX, valIDX) in enumerate(skf.split(n_sample, n_sample)):
    if idx == stp:
        break
    StartTime = time.time()
    fold_train_data = strat_train_set[trainIDX]
    fold_val_data = strat_train_set[valIDX]
    np.save(TARGET_FOLDER_PATH+"{}_{}_Batch{}_Eps{}_TRAIN_DATASET".format(Smoothing, Noise, batch_size, epochs), fold_train_data)
    np.save(TARGET_FOLDER_PATH+"{}_{}_Batch{}_Eps{}_VALID_DATASET".format(Smoothing, Noise, batch_size, epochs), fold_val_data)
    
    train_X = fold_train_data['Img'] / 255.0
    train_Y = fold_train_data['xH']
    val_X = fold_val_data['Img'] / 255.0
    val_Y = fold_val_data['xH']
    
    # Construct the learning algorithm
    ## CNN
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='valid', input_shape=(200,200,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(32, (3,3), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(64, (3,3), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2))
    model.add(Flatten())

    ## FC
    model.add(Dense(64, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(32, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('linear'))
    
    ## Set-up the Checkpoint
    MODEL_SAVE_FOLDER_PATH = TARGET_FOLDER_PATH+'CheckPointModels/'
    # TENSORBOARD_FOLDER_PATH = TARGET_FOLDER_PATH+'{}th_Tensorboard_Log/'.format(idx)
    if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
        os.makedirs(MODEL_SAVE_FOLDER_PATH, exist_ok=True)

    model_path = MODEL_SAVE_FOLDER_PATH + '{epoch:02d}-{val_loss:.4f}.hdf5'
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=0,
                                save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    ## Compile the model & train
    model.compile(loss='MSE', optimizer=opt, metrics=['MSE'])
    hist = model.fit(train_X, train_Y, epochs=epochs, verbose=2, batch_size=batch_size, validation_data=(val_X, val_Y), callbacks=callbacks_list, shuffle=True) 
    # When shuffle = True, only shuffle with training data

    ## Model Evaluation 
    scores = model.evaluate(val_X, val_Y, batch_size=batch_size, verbose=0)
    skfscores.append(scores[1])
    
    ## Save the history plot for fitting model and Save the final trained model
    print("Being drawing the graph of MODEL LOSS")
    plt.clf()
    plt.plot(hist.history['loss'], 'r',lw=1.7, label='train_loss')
    plt.plot(hist.history['val_loss'], 'b', lw=1.7, label='val_loss')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE LOSS')
    plt.legend(['train','validation'], loc='upper right')

    plt.savefig(TARGET_FOLDER_PATH+"{}_{}_Batch{}_Eps{}_LOSS.pdf".format(Smoothing, Noise, batch_size, epochs), bbox_inches="tight", dpi=150)
    plt.savefig(TARGET_FOLDER_PATH+"{}_{}_Batch{}_Eps{}_LOSS.png".format(Smoothing, Noise, batch_size, epochs), bbox_inches="tight", dpi=150)

    model.save(TARGET_FOLDER_PATH+"{}_{}_Batch{}_Eps{}_FinalModel.hdf5".format(Smoothing, Noise, batch_size, epochs))

    ## Record the evaluation scores for each k-fold. 
    ### Could look whether my own model has been likely to be overfitted for ONLY train data.
    EndTime = time.time()

    line = "{0}_th ::: LOSS = {1:0.5f}, MSE(METRIC) = {2:0.5f} \n".format(idx, scores[0], scores[1])
    timeline = "{0}_th ::: StartTime = {1:0.3f}, EndTime = {2:0.3f}, ElapsedTime = {3:0.3f}s \n".format(idx, StartTime, EndTime, (EndTime-StartTime))
    with open(TARGET_FOLDER_PATH+"{}_{}_Batch{}_Eps{}_SKFscores.txt".format(Smoothing, Noise, batch_size, epochs), 'a') as f:
        f.write(line)

Final_Time = time.time()
timeline = "Total Elapsed Time ::: {0:0.3f} \n".format((Final_Time-Initial_Time))
with open(TARGET_FOLDER_PATH+"{}_{}_Batch{}_Eps{}_SKFscores.txt".format(Smoothing, Noise, batch_size, epochs), 'a') as f:
    f.write(line)
    f.write(timeline)

# os.system("tar -cvf {}_{}_{}.tar {}_{}_{}".format(Smoothing, Noise, Color, Smoothing, Noise, Color))
    

