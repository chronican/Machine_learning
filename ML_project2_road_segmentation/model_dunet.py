from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, Activation, MaxPooling2D, Dropout, BatchNormalization
from keras.layers import add, concatenate
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
import keras.backend as K

from metrics import f1

def Conv2d_Block(x, n_filter,kernel_size=3, padding='same', activation='relu'):
    """ build a two-convolutional layer
    
    Args:
        input_shape {tuple}: size of the input image
        n_filter {int}: number of filter of the first layer 
        kernel_size {int or tuple}: the height and width of the 2D convolution window
        activation{string}: activation function to use
        
    Returns
    
    
    """
    x = Conv2D(n_filter, kernel_size=kernel_size, kernel_initializer="he_normal",padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(n_filter, kernel_size=kernel_size, kernel_initializer="he_normal",padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x


def DBlock(x, n_filter):
    """ build a Dilated Block 
    
    Args:
        x {tuple}: the input of the 
        n_filter {int}: number of filter of the first layer 
        
    Returns:
        tensor: output of the dilated block
    
    """
    dilate_1 = Conv2D(n_filter, kernel_size=3, dilation_rate=1, padding='same', activation='relu')(x)
    dilate_2 = Conv2D(n_filter, kernel_size=3, dilation_rate=2, padding='same', activation='relu')(dilate_1)
    dilate_3 = Conv2D(n_filter, kernel_size=3, dilation_rate=4, padding='same', activation='relu')(dilate_2)
    dilate_4 = Conv2D(n_filter, kernel_size=3, dilation_rate=8, padding='same', activation='relu')(dilate_3)
    out = add([ dilate_1, dilate_2, dilate_3, dilate_4])
    return out

def encoder(x,n_filter):
    """ build an encoder block
    
    Args:
        input_shape {tensor}: size of the input image
        n_filter {int}: number of filter of the first layer 
    
        
    Returns:
        tensor: output of the block
    
    """
    x=Conv2d_Block(x, n_filter,kernel_size=3, padding='same', activation='relu')
    x1=MaxPooling2D(pool_size=(2, 2))(x)
    return x,x1

def decoder(x,n_filter,encoder):
    """ build a two-convolutional layer
    
    Args:
        input_shape {tensor}: size of the input image
        n_filter {int}: number of filter of the first layer 
        encoder {tensor}: the input from the encoder block
        
    Returns:
        tensor: output of the block
    
    """
    
    dropout_rate=0.2
    x=Conv2DTranspose(n_filter, kernel_size=2, strides=2,kernel_initializer="he_normal", padding='same')(x)
    y=encoder
    x=concatenate([y, x], axis = 3)
    x=Dropout(dropout_rate)(x)
    x=Conv2D(n_filter, kernel_size=3, padding='same',kernel_initializer="he_normal")(x)
    x = Activation('relu')(x)
    x=Conv2D(n_filter, kernel_size=3, padding='same', kernel_initializer="he_normal")(x)
    x=Activation('relu')(x)
    return x


def unet_dilated(pretrained_weights=None,
                 input_size=(None, None, 3),
                 n_filter=16,
                 activation='relu',
                 dropout=True, dropout_rate=0.2,
                 batchnorm=True,
                 loss=binary_crossentropy,
                 optimizer=Adam(lr=1e-4)):
    '''The model of unet with dilated block
    
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
        keras.models -- UNet model with block
    '''

    inputs = Input(input_size)


    x1,encode1=encoder(inputs,n_filter)
    x2,encode2=encoder(encode1,n_filter * 2)
    x3,encode3=encoder(encode2,n_filter * 4)

    encode3=DBlock(encode3,n_filter * 8)

    decode1=decoder(encode3,n_filter * 4,x3)
    decode2=decoder(decode1,n_filter*2,x2)
    decode3=decoder(decode2,n_filter,x1)

    outputs=Conv2D(1, 1, activation='sigmoid')(decode3)
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=optimizer, loss=loss, metrics=[f1, 'accuracy'])

    if (pretrained_weights):
        model.load_weights(filepath=pretrained_weights)

    return model


