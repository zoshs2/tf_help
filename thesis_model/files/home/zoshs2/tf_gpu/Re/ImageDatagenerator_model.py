#!/home/zoshs2/anaconda3/envs/tf_gpu/bin/python
# ImageDatagenerator_model.py
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
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import argparse
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import StratifiedShuffleSplit

from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import time

basepath = '/home/zoshs2/tf_gpu/Re/DATA'
Color = 'RGB'
Noise = 'WG' # WG : WHITE_GAUSSIAN_NOISE , UNI : UNIFORM_NOISE
Smoothing = 'BS2_BW1'

TARGET_FOLDER_PATH = './{}_{}/'.format(Smoothing, Color)
if not os.path.exists(TARGET_FOLDER_PATH):
    os.makedirs(TARGET_FOLDER_PATH, exist_ok=True)

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
np.save(TARGET_FOLDER_PATH+"{}_{}_{}_TEST_DATASET".format(Smoothing, Noise, Color), strat_test_set)
del(strat_test_set)

np.random.shuffle(strat_train_set)
np.random.shuffle(strat_train_set)

# Stratified K Fold Cross Validation for evaluation to my learning algorithm in last.
n_splits=10
epochs = 200 # 상당히 좋은 200
random_state=4 # 상당히 좋은 2
stp=3
lr = 1e-3 # 상당히 좋은 1e-3
batch_size = 12 # 상당히 좋은 12 # 100했더니 너무 val이 크게 뭉탱이로 들쭉날쭉함 즉, 안예쁨.

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
skfscores = []
n_sample = np.zeros(strat_train_set.shape[0])
opt = Adam(lr=lr) # , decay=1e-3/100)
stp_point = stp # Stopping point for K-fold CV
times = 5 # 6000장쯤에서 3배 Aug.
StartTime = time.time()

for idx, (trainIDX, valIDX) in enumerate(skf.split(n_sample, n_sample)):
    if idx == stp_point:
        break
    train_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.5, height_shift_range=0.5, horizontal_flip=True, vertical_flip=True, fill_mode='wrap')
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    fold_train_data = strat_train_set[trainIDX]
    fold_val_data = strat_train_set[valIDX]
    
    train_generator = train_datagen.flow(fold_train_data['Img'], fold_train_data['xH'], batch_size=batch_size, shuffle=True)
    valid_generator = validation_datagen.flow(fold_val_data['Img'], fold_val_data['xH'], batch_size=batch_size, shuffle=True)
    
    # Construct the learning algorithm
    ## CNN
    print("{}_Processing the training model".format(idx))
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
    MODEL_SAVE_FOLDER_PATH = './{}_{}/{}th_CheckPointModels/'.format(Smoothing, Color, idx)
    TENSORBOARD_FOLDER_PATH = './{}_{}/{}th_Tensorboard_Log/'.format(Smoothing, Color, idx)
    if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
        os.makedirs(MODEL_SAVE_FOLDER_PATH, exist_ok=True)
    if not os.path.exists(TENSORBOARD_FOLDER_PATH):
        os.makedirs(TENSORBOARD_FOLDER_PATH, exist_ok=True)

    model_path = MODEL_SAVE_FOLDER_PATH + '{epoch:02d}-{val_loss:.4f}.hdf5'
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=2,
                                save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss', patience = 30)
    tb_hist = TensorBoard(log_dir=TENSORBOARD_FOLDER_PATH, histogram_freq=0, write_graph=True, batch_size=batch_size, write_images=True)
    callbacks_list = [checkpoint, tb_hist, early_stopping]

    ## Compile the model & train
    model.compile(loss='MSE', optimizer=opt, metrics=['MSE'])
    steps_per_epoch = int(fold_train_data['Img'].shape[0]/batch_size) * times
    ### steps_per_epoch : the number of traindata / batch_size * 데이터를 부풀리려 확보하게 될 데이터수 (e.g 2배, 3배, 4배 etc)
    hist = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, validation_data=valid_generator, epochs=epochs, callbacks=callbacks_list) 
    
    ## Model Evaluation 
    print("[INFO] Doing the Evaluation with Validation_Generator")
    scores = model.evaluate_generator(valid_generator, verbose=1)
    skfscores.append(scores[1])
    print("[INFO] Result from the evalution_generator is look like : ", scores)
    
    ## Save the history plot for fitting model and Save the final trained model
    print("[INFO] Being drawing the graph of MODEL LOSS")
    plt.clf()
    plt.plot(hist.history['loss'], 'r',lw=1.7, label='train_loss')
    plt.plot(hist.history['val_loss'], 'b', lw=1.7, label='val_loss')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE LOSS')
    plt.legend(['train','validation'], loc='upper right')
    
    plt.savefig(TARGET_FOLDER_PATH+"{}th_{}_{}_{}_MODEL_LOSS.pdf".format(idx, Smoothing, Noise, Color), bbox_inches="tight", dpi=150)
    plt.savefig(TARGET_FOLDER_PATH+"{}th_{}_{}_{}_MODEL_LOSS.png".format(idx, Smoothing, Noise, Color), bbox_inches="tight", dpi=150)
    
    model.save(TARGET_FOLDER_PATH+"{}th_{}_{}_{}_FinalModel_MSE.hdf5".format(idx, Smoothing, Noise, Color))
    
    ## Record the evaluation scores for each k-fold. 
    ### Could look whether my own model has been likely to be overfitted for ONLY train data.
    line = "{0}_th ::: LOSS = {1:0.5f}, MSE = {2:0.5f} \n".format(idx, scores[0], scores[1])
    with open(TARGET_FOLDER_PATH+"{}_{}_{}_SKFscores.txt".format(Smoothing, Noise, Color), 'a') as f:
        f.write(line)
    
skfscores_mean = np.mean(skfscores)
skfscores_std = np.std(skfscores)
EndTime = time.time()
ElapsedTime = EndTime - StartTime

line = "RESULT ::: K-fold Score :: MEAN = {0:0.5f}, STD = {1:0.5f} \n".format(skfscores_mean, skfscores_std)
timeline = "HOW MANY TIME HAD BEEN :: {}s".format(ElapsedTime)
with open(TARGET_FOLDER_PATH+"{}_{}_{}_SKFscores.txt".format(Smoothing, Noise, Color), 'a') as f:
    f.write(line)
    f.write(timeline)

os.system("tar -cvf {}_{}.tar {}_{}".format(Smoothing, Color, Smoothing, Color)) ## Allow HTcondor to send the whole output directory.

    

