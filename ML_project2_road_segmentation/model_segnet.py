# -*- coding: utf-8 -*-
"""

"""
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, Activation, MaxPooling2D, Dropout, BatchNormalization,UpSampling2D,Reshape
from keras.layers import add, concatenate,ZeroPadding2D
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

from metrics import f1


    

def con2d_2block(inputs, n_filter, kernel_size=3, activation='relu'):
    """ build a two-convolutional layer
    
    Args:
        input_shape {tensor}: size of the input image
        n_filter {int}: number of filter of the first layer 
        kernel_size {int or tuple}: the height and width of the filter (default: 3)
        activation{string}: activation function to use(default: 'relu')
        
    Returns:
        tensor: output of the block
    
    """
    conv1 = Conv2D(n_filter, kernel_size=kernel_size,padding='same')(inputs)
    x1 = BatchNormalization()(conv1)
    x2 = Activation(activation)(x1)
    conv2 = Conv2D(n_filter, kernel_size=kernel_size,padding='same')(x2)
    x3 = BatchNormalization()(conv2)
    x4 = Activation(activation)(x3)
    
    return x4
  
def con2d_3block(input_shape, n_filter, kernel_size=3, activation='relu'):
    
    """ build a three-convolutional layer
    
    Args:
        input_shape {tensor}: size of the input image
        n_filter {int}: number of filter of the first layer 
        kernel_size {int or tuple}: the height and width of the filter (default: 3)
        activation{string}: activation function to use(default: 'relu')
        
    Returns:
        tensor: output of the block
    
    """
    
    
    conv1 = Conv2D(n_filter, kernel_size=kernel_size,padding='same')(inputs)
    x1 = BatchNormalization()(conv1)
    x2 = Activation(activation)(x1)
    conv2 = Conv2D(n_filter, kernel_size=kernel_size,padding='same')(x2)
    x3 = BatchNormalization()(conv2)
    x4 = Activation(activation)(x3)
    conv3 = Conv2D(n_filter, kernel_size=kernel_size,padding='same')(x4)
    x5 = BatchNormalization()(conv3)
    x6 = Activation(activation)(x5)
    
    return x6
    


def segnet(
        pretrained_weights = None,
        input_shape = (None,None,3) ,
        kernel_size = (3,3),
        n_filter = 32,
        activation='relu',
        dropout=False, dropout_rate=0.5,
        loss=binary_crossentropy,
        optimizer=Adam(lr=1e-4)
        ):
    
    '''The model of segnet
    
    Arguments:
        pretrained_weights {str} : path of the pretrained weights (default: None)
        input_size {tuple}: size of input images (default: (None,None,3))
        n_filter {int} : number of filters used in the first layer (default: 32)
        activation {str}: activation function used in convolution layer(some activation layer may use other activation function) (default: 'relu')
        dropout {bool} : indicate variable of whether to add dropout layer (default: False)
        dropout_rate {float} : dropout rate  (default: 0.5)
        loss {keras.losses} : loss function to use (default: binary_crossentropy)
        optimizer {keras.optimizers}:optimizer to use in training model (default: Adam(lr=1e-4))
    
    Returns:
        keras.models -- SegNet model 
    
    '''
    model = Sequential()
    
    # encode layers
    model.add(Conv2D(n_filter, kernel_size=kernel_size,input_shape=input_shape,padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Conv2D(n_filter, kernel_size=kernel_size,padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))    
    model.add(MaxPooling2D(pool_size = (2,2)))
    
    
    
    model.add(Conv2D(n_filter*2, kernel_size=kernel_size,padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Conv2D(n_filter*2, kernel_size=kernel_size,padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))    
    model.add(MaxPooling2D(pool_size = (2,2)))
    
    
    model.add(Conv2D(n_filter*4, kernel_size=kernel_size,padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Conv2D(n_filter*4, kernel_size=kernel_size,padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Conv2D(n_filter*4, kernel_size=kernel_size,padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size = (2,2)))
   
    
    model.add(Conv2D(n_filter*8, kernel_size=kernel_size,padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Conv2D(n_filter*8, kernel_size=kernel_size,padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Conv2D(n_filter*8, kernel_size=kernel_size,padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size = (2,2)))
   
    # decode layers
    model.add(UpSampling2D())
    model.add(Conv2D(n_filter*8, kernel_size=kernel_size,padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Conv2D(n_filter*8, kernel_size=kernel_size,padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Conv2D(n_filter*8, kernel_size=kernel_size,padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    if dropout:
        model.add(Dropout(dropout_rate))
    
    
    
    model.add(UpSampling2D())
    model.add(Conv2D(n_filter*4, kernel_size=kernel_size,padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Conv2D(n_filter*4, kernel_size=kernel_size,padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Conv2D(n_filter*4, kernel_size=kernel_size,padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    if dropout:
        model.add(Dropout(dropout_rate))
    
    model.add(UpSampling2D())
    model.add(Conv2D(n_filter*2, kernel_size=kernel_size,padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Conv2D(n_filter*2, kernel_size=kernel_size,padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    if dropout:
        model.add(Dropout(dropout_rate))
    
    
    model.add(UpSampling2D())
    model.add(Conv2D(n_filter, kernel_size=kernel_size,padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Conv2D(1, kernel_size=(1,1),padding='valid'))
    model.add(BatchNormalization())
   
    
    model.add(Activation('sigmoid'))
    
    
    
    model.compile(optimizer = optimizer, loss = loss, metrics = [ f1,'accuracy'])
    
    if(pretrained_weights):
        model.load_weights(filepath=pretrained_weights)
    
    
    return model
    
  
    
    
    
    
    