import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras import models
from matplotlib.colors import LinearSegmentedColormap, Normalize


## Colorscheme reflecting to colorbars of feature map group.
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


class FeatureMap():
    '''
        Extract the feature maps passing through the Convolution layers 
        <<ORDER>>
        FeatureMap(<YOUR_MODEL>) ==> Create_Feature_Model() ==> Actuate_Feature_Model(Sample_Img) ==> Feature_Maps_Show()
    '''
    ## Colorscheme reflecting to colorbars of feature map group.
    cmap = LinearSegmentedColormap.from_list('mycmap', ['yellow','red','black','green','blue'])
    norm = MidpointNormalize(midpoint=0)
    
    def __init__(self, CNNmodel):
        '''
            예외처리 나중에 TODO
        '''
        self.__model = CNNmodel
        self.Feature_Model = False
        self.__Output_Layer_Flag = False
        self.__layer_outputs = None
        self.__layer_names = None
        self.__Activations = None
    
    def __Output_Layer(self):
        for idx, layer in enumerate(self.__model.layers):
            if "Flatten" in str(layer):
                break

        self.__layer_outputs = [ layer.output for layer in self.__model.layers[:idx]]
        self.__layer_names = [ layer.name for layer in self.__model.layers[:idx]]
        self.__Output_Layer_Flag = True
    
    def Create_Feature_Model(self):
        if not self.__Output_Layer_Flag:
            self.__Output_Layer()
            Activation_model = models.Model(inputs=self.__model.input, outputs=self.__layer_outputs)
            self.Feature_Model = Activation_model
    
    def Actuate_Feature_Model(self, Sample_Img):
        if not self.Feature_Model:
            return "Create_Feature_Model을 먼저 하세요."
        if len(Sample_Img.shape) != 4:
            print(f"Sample Img shape must be (1, ?, ?, ?) but is {Sample_Img.shape}. \n np.expand_dims(Sample_Img, axis=0) may solve this problem")
            return
        Activations = self.Feature_Model.predict(Sample_Img)
        self.__Activations = Activations
    
    def Feature_Maps_Show(self, savefig=None, n_cols=8, cmap=cmap, norm=norm):
        '''
            우선 Conv를 지나고 바로 직후의 Feature Maps를 뽑아내도록 만들었음.
            (Pooling, Batch_Norm, Activation_Func 바로 직후는 볼 수 없음)
        '''
        n_cols = n_cols
        
        for layer_name, layer_activation in zip(self.__layer_names, self.__Activations): # Displays the feature maps
            if 'conv' not in layer_name:
                continue
            
            n_features = layer_activation.shape[-1] # Number of features in the feature map
            if n_features % n_cols != 0:
                n_rows = int(n_features/n_cols) + 1
            else: n_rows = n_features // n_cols # Tiles the activation channels in this matrix

            fig, axe = plt.subplots(n_rows, n_cols, figsize=(15, 15))
            fig.suptitle(layer_name, fontsize=30)
            idx = 1
            itr_checkpoint = 1
            for ax_row in axe:
                for ax_col in ax_row:
                    if itr_checkpoint == n_features+1:
                        break
                    ax_col.set_xticks([])
                    ax_col.set_yticks([])
                    vmin = np.percentile(layer_activation[0,:,:,idx-1], 30)
                    vmax = np.percentile(layer_activation[0,:,:,idx-1], 100)
                    ax_col.imshow(layer_activation[0,:,:,idx-1], vmax=vmax, vmin=vmin, cmap=cmap, norm=norm)
                    idx += 1
                    itr_checkpoint += 1
            fig.tight_layout()
            fig.subplots_adjust(top=0.88)
            if savefig:
                 plt.savefig(savefig+".pdf", dpi=150, bbox_inches="tight", pad_inches=0)
            plt.show()
            
        return 