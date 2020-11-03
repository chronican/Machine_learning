# -*- coding: utf-8 -*-
"""

"""

from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, Activation, MaxPooling2D, Dropout, BatchNormalization, \
    LeakyReLU
from keras.layers import add, concatenate
from keras.losses import binary_crossentropy
from keras.optimizers import Adam

from metrics import f1


def Conv2d_Block(x, n_filter, kernel_size, strides=(1, 1), padding='same', activation='relu'):
    """ Build a convolutional block  with batchnormalization and activation
    
    Arguments:
        x {tensor}:the input of the block
        n_filter {int}: number of input of the 
        strides {tuple}: strides of the convolution layer
        padding {str}: padding scheme of the convolution layer
        activation {str}: activation scheme
        
    Return:
        tensor : the output of this convolution block    
    """
    x = Conv2D(n_filter, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation(activation)(x)
    return x


def Conv2dT_Block(x, n_filter, kernel_size, strides=(2, 2), padding='same', activation='relu'):
    """ Build a convolutional block  with batchnormalization and activation
    
    Arguments:
        x {tensor}:the input of the block
        n_filter {int}: number of input of the 
        strides {tuple}: strides of the convolution layer
        padding {str}: padding scheme of the convolution layer
        activation {str}: activation scheme
        
    Return:
        tensor : the output of this convolution block    
    """
    x = Conv2DTranspose(n_filter, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation(activation)(x)
    return x
    
     
   


def ResBlock(x, n_filter, first_layer_strides = (1,1),is_add = False):
    """ Build a convolutional transpose block  with batchnormalization and activation
    
    Arguments:
        x {array}:the input of the block
        n_filter {int}: number of input of the 
        strides {tuple}: strides of the convolution transpose layer
        padding {str}: padding scheme of the convolution transpose layer
        activation {str}: activation scheme
        
    Return:
        x {array}: the output of this convolution transpose block    
    """
    if is_add:
        conv1 = Conv2d_Block(x, n_filter, kernel_size=3, strides = first_layer_strides, padding='same')
        conv2 = Conv2d_Block(conv1, n_filter, kernel_size=3, strides=(1,1), padding='same')  
        #conv3 = Conv2d_Block(x, n_filter, kernel_size=1, strides=(1, 1), padding='same')    
        out = add([x,conv2])
        
    else:
        conv1 = Conv2d_Block(x, n_filter, kernel_size=3, strides = first_layer_strides, padding='same')
        conv2 = Conv2d_Block(conv1, n_filter, kernel_size=3, strides=(1, 1), padding='same') 
        out = conv2
        
    '''  
    conv1 = Conv2d_Block(x, n_filter, kernel_size=3, strides=(1, 1), padding='same')
    conv2 = Conv2d_Block(conv1, n_filter, kernel_size=3, strides=(1, 1), padding='same')  
    conv3 = Conv2d_Block(x, n_filter, kernel_size=1, strides=(1, 1), padding='same')    
    out = add([conv3,conv2])
    '''
    return out 

def Encoder_3Res(x,n_filter):
    ''' Concatenation of 3 Res-block  '''
    x = ResBlock(x, n_filter, first_layer_strides = (2,2),is_add = False)
    x = ResBlock(x, n_filter, first_layer_strides = (1,1),is_add = True)
    x = ResBlock(x, n_filter, first_layer_strides = (1,1),is_add = True)
    
    #x = MaxPooling2D(pool_size = (2,2))(x)
    
    return x

def Encoder_4Res(x,n_filter): 
    ''' Concatenation of 4 Res-block  '''
    x = ResBlock(x, n_filter, first_layer_strides = (2,2),is_add = False)
    x = ResBlock(x, n_filter, first_layer_strides = (1,1),is_add = True)
    x = ResBlock(x, n_filter, first_layer_strides = (1,1),is_add = True) 
    x = ResBlock(x, n_filter, first_layer_strides = (1,1),is_add = True)
    
    #x = MaxPooling2D(pool_size = (2,2))(x)
    
    return x

def Encoder_6Res(x,n_filter):
    ''' Concatenation of 6 Res-block  '''
    x = ResBlock(x, n_filter, first_layer_strides = (2,2),is_add = False)
    x = ResBlock(x, n_filter, first_layer_strides = (1,1),is_add = True)
    x = ResBlock(x, n_filter, first_layer_strides = (1,1),is_add = True) 
    x = ResBlock(x, n_filter, first_layer_strides = (1,1),is_add = True)
    x = ResBlock(x, n_filter, first_layer_strides = (1,1),is_add = True) 
    x = ResBlock(x, n_filter, first_layer_strides = (1,1),is_add = True)
    
    #x = MaxPooling2D(pool_size = (2,2))(x)
     
    
    return x

def Decoder(x, m_filter, n_filter):
    x = Conv2d_Block(x, m_filter // 4, kernel_size=1, strides=(1, 1), padding='same')
    x = Conv2dT_Block(x, m_filter // 4, kernel_size=3, strides=(2, 2), padding='same')
    x = Conv2d_Block(x, n_filter, kernel_size=1, strides=(1, 1), padding='same')
    return x

def DBlock(x,n_filter):
    """ build a decode block
    
    Args:
        x {tensor}: the input of the image
        m_filter {int}: number of filter of the first two layer        
        n_filter {int}: number of filter of the last layer 
        
    Returns:
        tensor: output of the block
    
    """
    dilate_1 = Conv2D(n_filter, kernel_size=3, dilation_rate=1, padding='same',activation='relu')(x)
    dilate_2 = Conv2D(n_filter, kernel_size=3, dilation_rate=2, padding='same',activation='relu')(dilate_1)
    dilate_3 = Conv2D(n_filter, kernel_size=3, dilation_rate=4, padding='same',activation='relu')(dilate_2)
    dilate_4 = Conv2D(n_filter, kernel_size=3, dilation_rate=8, padding='same',activation='relu')(dilate_3)
    dilate_5 = Conv2D(n_filter, kernel_size=3, dilation_rate=16, padding='same',activation='relu')(dilate_4)
    out = add([x,dilate_1,dilate_2,dilate_3,dilate_4,dilate_5])
    return out

def DLinknet(pretrained_weights=None,
            input_size=(None, None, 3),
            n_filter=32,
            activation='relu',
            dropout=True, dropout_rate=0.5,
            batchnorm=True,
            loss=binary_crossentropy,
            optimizer=Adam(lr=1e-4)):
    """Build a standard DLinknet model.

    Arguments:
        pretrained_weights {str} -- path of the pretrained weights (default: {None})
        input_size {tuple} -- size of input images (default: {(None,None,3)})
        n_filter {int} -- number of filter of the first layer (default: {32})
        activation {str} -- activation function to use (default: {'relu'})
        dropout {bool} -- whether to use dropout layer (default: {True})
        dropout_rate {float} -- dropout rate (default: {0.5})
        batchnorm {bool} -- whether to use batch normalization layer (default: {True})
        loss {keras.losses} -- loss function to use (default: {binary_crossentropy})
        optimizer {keras.optimizers} -- optimizer to use (default: {Adam(lr=1e-4)})

    Returns:
        keras.models -- Dlinknet model
    """

    # 3
    inputs = Input(input_size)


    conv1 = Conv2d_Block(inputs, n_filter, kernel_size=7, strides=(2, 2))

    # encoding part: resnet34
    encode1 = Encoder_3Res(conv1, n_filter)
    encode2 = Encoder_4Res(encode1, n_filter*2)
    encode3 = Encoder_6Res(encode2, n_filter*4)
    encode4 = Encoder_3Res(encode3, n_filter*8)
    
    # dilated block 
    encode4 = DBlock(encode4,n_filter= n_filter*8)

    # decode part 
    decode1=Decoder(encode4, n_filter*8,n_filter*4)
    decode1= add([decode1,encode3])
    decode2 = Decoder(encode3, n_filter*4, n_filter*2)
    decode2 = add([decode2, encode2])
    decode3 = Decoder(decode2, n_filter*2, n_filter)
    decode3 = add([decode3, encode1])
    decode4 = Decoder(decode3, n_filter, n_filter)
    # decode4=add([decode4,encode1])

    fullconv1 = Conv2dT_Block(decode4, 32, kernel_size=2, strides=(2, 2), padding='same')
    conv2 = Conv2d_Block(fullconv1, 32, kernel_size=3, strides=(1, 1))
    fullconv2 = Conv2dT_Block(conv2, 1, kernel_size=1, strides=(1, 1), padding='same', activation='sigmoid')

    model = Model(inputs=inputs, outputs=fullconv2)

    model.compile(optimizer=optimizer, loss=loss, metrics=[f1, 'accuracy'])

    if (pretrained_weights):
        model.load_weights(filepath=pretrained_weights)

    return model