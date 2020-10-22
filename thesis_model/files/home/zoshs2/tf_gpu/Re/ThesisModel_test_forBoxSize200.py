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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import StratifiedShuffleSplit
import time


epochs = 500 
batch_size = 8 # ['4', 8, 16, 32, 64, 128, 256] 

# Target Configuration
basepath = '/home/zoshs2/tf_gpu/Re/DATA'
Box_Size = 200 # [ 50, 100, 200, 300]
# Color = 'RGB' # To date, this is only existed. - RGB
Noise = 'WG' # WG : WHITE_GAUSSIAN_NOISE , UNI : UNIFORM_NOISE, NN : Without Nowise
Smoothing = 'BS2BW1' # BS{a}BW{b} where a, b are BEAMSIZE and BANDWIDTH that we had convolved, respectively. 
Obs_Time = '1000h' 

# Save Configuration
TARGET_FOLDER_PATH = './BoxSize{}_{}_{}_Batch{}_Eps{}/'.format(Box_Size, Smoothing, Noise, batch_size, epochs)
if not os.path.exists(TARGET_FOLDER_PATH):
    os.makedirs(TARGET_FOLDER_PATH, exist_ok=True)

# Targeting Whole Dataset Load
data = np.load("{}/Box{}_{}_{}_{}.npy".format(basepath, Box_Size, Smoothing, Noise, Obs_Time))

# Data Split into Test and Train.
X_train, X_test, y_train, y_test = train_test_split(data['Img'], data['xH'], test_size=0.20)
del(data)

# Save the test dataset for inference after training.
np.save(TARGET_FOLDER_PATH+"Box{}_{}_{}{}_Batch{}_Eps{}_X_TEST_DATASET".format(Box_Size, Smoothing, Noise, Obs_Time, batch_size, epochs), X_test)
np.save(TARGET_FOLDER_PATH+"Box{}_{}_{}{}_Batch{}_Eps{}_Y_TEST_DATASET".format(Box_Size, Smoothing, Noise, Obs_Time, batch_size, epochs), y_test)
del(X_test)
del(y_test)
print("TESTSET SAVE!")

## 
lr = 1e-3 # 상당히 좋은 1e-3
skfscores = []
opt = Adam(lr=lr) # , decay=1e-3/100) 

StartTime = time.time()

np.save(TARGET_FOLDER_PATH+"Box{}_{}_{}{}_Batch{}_Eps{}_X_TRAIN_DATASET".format(Box_Size, Smoothing, Noise, Obs_Time, batch_size, epochs), X_train)
np.save(TARGET_FOLDER_PATH+"Box{}_{}_{}{}_Batch{}_Eps{}_Y_TRAIN_DATASET".format(Box_Size, Smoothing, Noise, Obs_Time, batch_size, epochs), y_train)
    
train_X = X_train / 255.0
train_Y = y_train
    
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
hist = model.fit(train_X, train_Y, epochs=epochs, verbose=2, batch_size=batch_size, validation_split=0.2, callbacks=callbacks_list, shuffle=True) 
# When shuffle = True, only shuffle with training dat
    
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

plt.savefig(TARGET_FOLDER_PATH+"Box{}_{}_{}{}_Batch{}_Eps{}_LOSS.pdf".format(Box_Size, Smoothing, Noise, Obs_Time, batch_size, epochs), bbox_inches="tight", dpi=150)
plt.savefig(TARGET_FOLDER_PATH+"Box{}_{}_{}{}_Batch{}_Eps{}_LOSS.png".format(Box_Size, Smoothing, Noise, Obs_Time, batch_size, epochs), bbox_inches="tight", dpi=150)

model.save(TARGET_FOLDER_PATH+"Box{}_{}_{}{}_Batch{}_Eps{}_FinalModel.hdf5".format(Box_Size, Smoothing, Noise, Obs_Time, batch_size, epochs))

## Record the evaluation scores for each k-fold. 
### Could look whether my own model has been likely to be overfitted for ONLY train data.

Final_Time = time.time()
timeline = "Total Elapsed Time ::: {0:0.3f}min \n".format((Final_Time-StartTime)/60)
with open(TARGET_FOLDER_PATH+"Box{}_{}_{}{}_Batch{}_Eps{}_train_info.txt".format(Box_Size, Smoothing, Noise, Obs_Time, batch_size, epochs), 'a') as f:
    f.write(timeline)

# os.system("tar -cvf {}_{}_{}.tar {}_{}_{}".format(Smoothing, Noise, Color, Smoothing, Noise, Color))
    

