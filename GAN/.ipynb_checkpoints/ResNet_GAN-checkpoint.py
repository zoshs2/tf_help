#!/home/zoshs2/anaconda3/envs/tf_gpu/bin/python
# ResNet_GAN.py

import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model, Sequential
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import time
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.multiclass import type_of_target
import os 

def identity_block(X, f, filters):
    """
    Implementation of the identity block as defined in Figure 3
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def convolutional_block(X, f, filters, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), padding="same", kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)


    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization()(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

class GAN():
    def __init__(self):
        self.img_rows = 200
        self.img_cols = 200
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.discriminator_model()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.generator_model_ResNet50()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(self.img_shape)
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)
        
        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    def generator_model_ResNet50(self):
        """
        Implementation of the popular ResNet50 the following architecture:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL(X) -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL(X) -> TOPLAYER

        Arguments:
        input_shape -- shape of the images of the dataset
        classes -- integer, number of classes

        Returns:
        model -- a Model() instance in Keras
        """

        # Define the input as a tensor with shape input_shape
        X_input = Input(self.img_shape)

        # Zero-Padding
        X = ZeroPadding2D((3, 3))(X_input)

        # Stage 1
        X = Conv2D(64, (7, 7), strides=(1, 1), kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        # Stage 2
        X = convolutional_block(X, f=3, filters=[64, 64, 256], s=1)
        X = identity_block(X, 3, [64, 64, 256])
        X = identity_block(X, 3, [64, 64, 256])
        
        ### START CODE HERE ###

        # Stage 3 (≈4 lines)
        X = convolutional_block(X, f = 3, filters = [128, 128, 512], s = 1)
        X = identity_block(X, 3, [128, 128, 512])
        X = identity_block(X, 3, [128, 128, 512])
        X = identity_block(X, 3, [128, 128, 512])
        
        # Stage 4 (≈6 lines)
        X = convolutional_block(X, f = 3, filters = [256, 256, 1024], s = 1)
        X = identity_block(X, 3, [256, 256, 1024])
        X = identity_block(X, 3, [256, 256, 1024])
        X = identity_block(X, 3, [256, 256, 1024])
        X = identity_block(X, 3, [256, 256, 1024])
        X = identity_block(X, 3, [256, 256, 1024])
        
        # Stage 5 (≈3 lines)
        X = convolutional_block(X, f = 3, filters = [512, 512, 2048], s = 1)
        X = identity_block(X, 3, [512, 512, 2048])
        X = identity_block(X, 3, [512, 512, 2048])
        
        # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
        # X = AveragePooling2D((2,2), name="avg_pool")(X)

        ### END CODE HERE ###
        # output layer
        X = Conv2D(3, (3, 3), padding='same', kernel_initializer=glorot_uniform(seed=0))(X) 
            # 여기서 Conv2D 의 the amount of filters 는 채널갯수!! RGB는 3개 , GRAY는 1개, RGBA는 4개 겠지.
        X = BatchNormalization()(X)
        gen_img = Activation('sigmoid')(X)
        
        # Create model
        model = Model(inputs=X_input, outputs=gen_img, name='Generator_Model_ResNet50')
        return model
    
    def discriminator_model(self):
        
        X_input = Input(shape=self.img_shape)
        
        x = Conv2D(32, (3,3), padding='valid', input_shape=(self.img_shape), kernel_initializer=glorot_uniform(seed=0))(X_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

        x = Conv2D(32, (3,3), padding='valid', kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

        x = Conv2D(64, (3,3), padding='valid', kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2,2), strides=2)(x)
        x = Flatten()(x)

        ## FC
        x = Dense(64, kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(32, kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=X_input, outputs=x, name='Discriminator')

        return model
    
    def train(self, blur_arr, clean_arr, epochs=100, batch_size=128):
        # Load the dataset
        X_train = blur_arr
        Y_train = clean_arr
        half_batch = int(batch_size / 2)
        
        for epoch in range(epochs):
            ######################
            # Train Discriminator
            ######################
            
            idx = np.random.randint(0, Y_train.shape[0], half_batch)
            clean_imgs = Y_train[idx]
            blurred_imgs = X_train[idx]
            gen_imgs = self.generator.predict(blurred_imgs)
            
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(clean_imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            ######################
            # Train Generator
            ######################
            valid_y = np.array([1] * batch_size)
            
            # Train the generator
            g_loss = self.combined.train_on_batch(blurred_imgs, valid_y)
            
            
            # Print the progress
            line = "%d [D loss: %f, acc.: %.2f%%] [G loss: %f] \n" % (epoch, d_loss[0], 100*d_loss[1], g_loss)
            
            with open(TARGET_FOLDER_PATH+"{}_Deblur_GAN_Loss_and_Accuracy.txt".format(Smoothing), 'a') as f:
                f.write(line)
            
    
if __name__ == '__main__':
    
    basepath = '/home/zoshs2/tf_gpu/Re/DATA'
    Smoothing = 'BS1_BW1'
    Random_Seed = 5 # Random Seed for Shuffling the training dataset

    TARGET_FOLDER_PATH = './{}_deblurGAN/'.format(Smoothing)
    if not os.path.exists(TARGET_FOLDER_PATH):
        os.makedirs(TARGET_FOLDER_PATH, exist_ok=True)

    data_blurred = np.load("{}/{}_NN_RGB_10318.npy".format(basepath, Smoothing))
    data_origin = np.load("{}/BS0_BW0_NN_RGB_10318.npy".format(basepath))

    dtype = [('Img','i4',(200,200,3)),('xH','f4'),('Box','i4'),('z','f4'),('index','i4'),('xH_cat', 'i4')]
    Blurred_data = np.empty(data_blurred.shape[0], dtype=dtype)
    Origin_data = np.empty(data_origin.shape[0], dtype=dtype)

    arr_cat = np.int32(np.ceil(data_origin['xH'] / 0.01))

    Blurred_data['Img'] = data_blurred['Img']
    Origin_data['Img'] = data_origin['Img']

    Blurred_data['xH'] = data_blurred['xH']
    Origin_data['xH'] = data_origin['xH']

    Blurred_data['Box'] = data_blurred['Box']
    Origin_data['Box'] = data_origin['Box']

    Blurred_data['z'] = data_blurred['z']
    Origin_data['z'] = data_origin['z']

    Blurred_data['index'] = data_blurred['index']
    Origin_data['index'] = data_origin['index']

    Blurred_data['xH_cat'] = arr_cat
    Origin_data['xH_cat'] = arr_cat
    del(data_blurred) # DON'T WASTE MEMORIES
    del(data_origin) # DON'T WASTE MEMORIES

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=21)
    for train_index, test_index in split.split(Origin_data, Origin_data['xH_cat']):
        Origin_train_set = Origin_data[train_index]
        Origin_test_set = Origin_data[test_index]
        Blurred_train_set = Blurred_data[train_index]
        Blurred_test_set = Blurred_data[test_index]

    del(Blurred_data) # DON'T WASTE MEMORIES
    del(Origin_data) # DON'T WASTE MEMORIES

    # Save the test dataset for inference after training.
    np.save(TARGET_FOLDER_PATH+"{}_NN_TEST_DATASET.npy".format(Smoothing), Origin_test_set)
    np.save(TARGET_FOLDER_PATH+"{}_NN_TEST_DATASET.npy".format(Smoothing), Blurred_test_set)
    del(Origin_test_set) # DON'T WASTE MEMORIES
    del(Blurred_test_set) # DON'T WASTE MEMORIES

    np.random.seed(Random_Seed)
    np.random.shuffle(Origin_train_set)
    np.random.seed(Random_Seed)
    np.random.shuffle(Blurred_train_set)

    # Stratified K Fold Cross Validation for evaluation to my learning algorithm in last.
    n_splits = 10
    epochs = 200 # 
    random_state = 2 # 
    stp=1
    # lr = 1e-3 # 상당히 좋은 1e-3
    batch_size = 128 # 상당히 좋은 12
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    skfscores = []
    n_sample = np.zeros(Origin_train_set.shape[0])
    # opt = Adam(lr=lr) # , decay=1e-3/100)
    stp_point = stp # Stopping point for K-fold CV
    Initial_Time = time.time()

    for idx, (trainIDX, valIDX) in enumerate(skf.split(n_sample, n_sample)):
        if idx == stp_point:
            break
        
        Initial_Time = time.time()
       
        y_train_data = Origin_train_set[trainIDX]
        x_train_data = Blurred_train_set[trainIDX]
        y_val_data = Origin_train_set[valIDX]
        x_val_data = Blurred_train_set[valIDX]

        train_X = x_train_data['Img'] / 255.0
        train_Y = y_train_data['Img'] / 255.0
        val_X = x_val_data['Img'] / 255.0
        val_Y = y_val_data['Img'] / 255.0
    
        gan = GAN()
        gan.train(train_X, train_Y, epochs=epochs, batch_size=batch_size)
        
        FinalTime = time.time()
        timeline = "Total Elapsed Time ::: {0:0.3f} \n".format((Final_Time-Initial_Time))
        with open(TARGET_FOLDER_PATH+"{}_Deblur_GAN_Loss_and_Accuracy.txt".format(Smoothing), 'a') as f:
            f.write(timeline)
            
os.system("tar -cvf {}_deblurGAN.tar {}_deblurGAN".format(Smoothing, Smoothing))
