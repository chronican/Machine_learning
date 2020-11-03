#!/usr/bin/env python3

from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, Activation, MaxPooling2D, Dropout, BatchNormalization
from keras.layers import add, concatenate, RepeatVector,Add,Multiply
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.backend import tile

from metrics import f1

def conv2d_block(inputs, n_filter, kernel_size=3, activation='relu'):
    """ build a two-convolutional layer
    
    Args:
        inputs {tensor}: input of the block
        n_filter {int}: number of filter of the first layer 
        kernel_size {int or tuple}: the height and width of the 2D convolution window
        activation{string}: activation function to use
        
    Returns:
        tensor : the output of the block 

    """
    # first layer
    x = Conv2D(n_filter, kernel_size=kernel_size, kernel_initializer="he_normal",
               padding="same")(inputs)
    
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(n_filter, kernel_size=kernel_size, kernel_initializer="he_normal",
              padding="same")(x)

    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x

def Conv2d_Block(inputs, n_filter, kernel_size=3, activation='relu'):
    """ build a two-convolutional layer
    
    Args:
        inputs {tuple}: input of the block
        n_filter {int}: number of filter of the first layer 
        kernel_size {int or tuple}: the height and width of the 2D convolution window
        activation{string}: activation function to use
        
    Returns
        tensor: output of the block
    
    """
    
    x = Conv2D(n_filter, kernel_size=kernel_size, kernel_initializer="he_normal",
               padding="same")(inputs)
    x = Activation(activation)(x)
    x = Conv2D(n_filter, kernel_size=kernel_size, kernel_initializer="he_normal",
              padding="same")(x)
    x = Activation(activation)(x)
    
    return x

def AG(input1,input2,dim):
    """ build an attention gate 
    
    Args:
        input_shape {tuple}: size of the input image
        n_filter {int}: number of filter of the first layer 
        kernel_size {int or tuple}: the height and width of the 2D convolution window
        activation{string}: activation function to use
        
    Returns
        tensor: output of the block
    
    """
    x_1 = Conv2D(dim, kernel_size=1, strides=(1,1),kernel_initializer="he_normal",activation=None, use_bias=True)(input1)
    x_1 = BatchNormalization()(x_1)
    x_2 = Conv2D(dim, kernel_size=1, strides=(1,1),kernel_initializer="he_normal",activation=None, use_bias=True)(input2)
    x_2 = BatchNormalization()(x_2)
    x_3 = Add()([x_1,x_2])
    x_3 = Activation('relu')(x_3)
    x_3 = Conv2D(1, kernel_size=1, strides=(1,1),kernel_initializer="he_normal",activation=None, use_bias=True)(x_3)
    x_3 = BatchNormalization()(x_3)
    x_3 = Activation('sigmoid')(x_3)
    x_4 = x_3
    for i in range(dim-1):
        x_4 = concatenate([x_4, x_3], axis = 3)
    x_5 = Multiply()([input1,x_4])

    return x_5

def unet_Attention(pretrained_weights = None,
         input_size = (None,None,3),
         n_filter=16,
         activation='relu',
         dropout=False, dropout_rate=0.2,
         batchnorm=True,
         loss=binary_crossentropy,
         optimizer=Adam(lr=1e-4)):
    '''The model of U-Net with attention
    
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
        keras.models -- U-Net model with attention
    
    '''

  
    # 3
    inputs = Input(input_size)
    
    # down path
    # n_filter
    conv1 = conv2d_block(inputs, n_filter, kernel_size=3, activation=activation)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # n_filter*2
    conv2 = conv2d_block(pool1, n_filter*2, kernel_size=3, activation=activation)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # n_filter*4
    conv3 = conv2d_block(pool2, n_filter*4, kernel_size=3, activation=activation)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # n_filter*8
    conv4 = conv2d_block(pool3, n_filter*8, kernel_size=3, activation=activation)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # central path
    # n_filter*16
    conv5 = conv2d_block(pool4, n_filter*16, kernel_size=3,  activation=activation)
    
    up6 = Conv2DTranspose(n_filter*8, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(conv5)
    conv4 = AG(conv4,up6,n_filter*8)
    merge6 = concatenate([conv4, up6], axis = 3)
    merge6 = Dropout(dropout_rate)(merge6)
    conv6 = Conv2d_Block(merge6, n_filter*8, kernel_size=3, activation=activation)
    
    # n_filter*4
    up7 = Conv2DTranspose(n_filter*4, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(conv6)
    conv3 = AG(conv3,up7,n_filter*4)
    merge7 = concatenate([conv3, up7], axis = 3)
    merge7 = Dropout(dropout_rate)(merge7) 
    conv7 = Conv2d_Block(merge7, n_filter*4, kernel_size=3, activation=activation)
    
    # n_filter*2
    up8 = Conv2DTranspose(n_filter*2, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(conv7)
    conv2 = AG(conv2,up8,n_filter*2)
    merge8 = concatenate([conv2, up8], axis = 3)
    merge8 = Dropout(dropout_rate)(merge8)
    conv8 = Conv2d_Block(merge8, n_filter*2, kernel_size=3, activation=activation)
    
    # n_filter
    up9 = Conv2DTranspose(n_filter, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(conv8)
    conv1 = AG(conv1,up9,n_filter)
    merge9 = concatenate([conv1, up9], axis = 3)
    merge9 = Dropout(dropout_rate)(merge9) 
    conv9 =Conv2d_Block(merge9, n_filter, kernel_size=3, activation=activation)
    
    # classifier
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = optimizer, loss = loss, metrics = [f1, 'accuracy'])
    
    if(pretrained_weights):
        model.load_weights(filepath=pretrained_weights)

    return model



