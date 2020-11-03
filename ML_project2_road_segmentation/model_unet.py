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
        x {tensor}: the input of the block
        n_filter {int}: number of filter of the first layer 
        kernel_size {int or tuple}: the height and width of the filter (default: 3)
        padding{string}: the padding scheme of the convolutional block(default: 'same')
        activation{string}: activation function to use(default: 'relu')
        
    Returns:
        tensor: output of the block
    
    """
    x = Conv2D(n_filter, kernel_size, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(n_filter, kernel_size, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x


def encoder(x,n_filter):
    """ build an encoder block
    
    Args:
        x {tensor}: the input of this block
        n_filter {int}: number of filter of the first layer 
    
        
    Returns:
        tensor: output of the block
    
    """
    x=Conv2d_Block(x, n_filter,kernel_size=3, padding='same', activation='relu')
    x=MaxPooling2D(pool_size=(2, 2))(x)
    return x

def decoder(x,n_filter,encoder):
    """ build a two-convolutional layer
    
    Args:
        x {tensor}: the input of the block
        n_filter {int}: number of filter of the first layer 
        encoder {tensor}: the input from the encoder block
        
    Returns:
        tensor: output of the block
    
    """
    dropout_rate=0.2
    x=Conv2DTranspose(n_filter, kernel_size=2, strides=2,kernel_initializer="he_normal", padding='same')(x)
    y=Conv2d_Block(encoder,n_filter,kernel_size=3, padding='same', activation='relu')
    x=concatenate([y, x], axis = 3)
    x=Dropout(dropout_rate)(x)
    x=Conv2D(n_filter, kernel_size=3, padding='same',activation='relu')(x)
    x=Conv2D(n_filter, kernel_size=3, padding='same', activation='relu')(x)
    return x

def unet(pretrained_weights = None,
         input_size = (None,None,3),
         n_filter=32,
         activation='relu',
         dropout=True, dropout_rate=0.2,
         batchnorm=True,
         loss=binary_crossentropy,
         optimizer=Adam(lr=1e-4)):
    '''The model of unet
    
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
        keras.models -- UNet model 
    
    '''

    inputs = Input(input_size)

    encode1=encoder(inputs,n_filter)
    encode2=encoder(encode1,n_filter*2)
    encode3=encoder(encode2, n_filter * 4)
    encode4=encoder(encode3,n_filter*8)

    encode5=Conv2d_Block(encode4, n_filter*16,kernel_size=3, padding='same', activation='relu')

    decode1=decoder(encode5,n_filter*8,encode3)
    decode2=decoder(decode1,n_filter*4,encode2)
    decode3=decoder(decode2,n_filter*2,encode1)
    decode4=decoder(decode3,n_filter,inputs)

    output=Conv2D(1, 1, activation='sigmoid')(decode4)
    model = Model(inputs=inputs, outputs=output)

    model.compile(optimizer=optimizer, loss=loss, metrics=[f1, 'accuracy'])

    if (pretrained_weights):
        model.load_weights(filepath=pretrained_weights)

    return model





