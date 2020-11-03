

from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, Activation, MaxPooling2D, Dropout, BatchNormalization, \
    LeakyReLU
from keras.layers import add, concatenate ,Add,Multiply
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


def Encoder(x, n_filter):
    """ build an encoder block
    
    Args:
        input_shape {tensor}: size of the input image
        n_filter {int}: number of filter of the first layer 
    
        
    Returns:
        tensor: output of the block
    """
    rec1 = x
    # if(!(m_filter==n_filter)):
    rec1 = Conv2d_Block(rec1, n_filter, kernel_size=1, strides=(2, 2), padding='same')
    x = Conv2d_Block(x, n_filter, kernel_size=3, strides=(2, 2), padding='same')
    x = Conv2d_Block(x, n_filter, kernel_size=3, strides=(1, 1), padding='same')
    x = add([x, rec1])
    rec2 = x
    x = Conv2d_Block(x, n_filter, kernel_size=3, strides=(1, 1), padding='same')
    x = Conv2d_Block(x, n_filter, kernel_size=3, strides=(1, 1), padding='same')
    x = add([x, rec2])
    return x


def Decoder(x, m_filter, n_filter):
    """ build a decode block
    
    Args:
        x {tensor}: the input of the image
        m_filter {int}: number of filter of the first two layer        
        n_filter {int}: number of filter of the last layer 
        
    Returns:
        tensor: output of the block
    
    """
    x = Conv2d_Block(x, m_filter // 4, kernel_size=1, strides=(1, 1), padding='same')
    x = Conv2dT_Block(x, m_filter // 4, kernel_size=2, strides=(2, 2), padding='same')
    x = Conv2d_Block(x, n_filter, kernel_size=1, strides=(1, 1), padding='same')
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
    x_1 = Conv2D(dim, kernel_size=1, strides=(1,1),kernel_initializer="he_normal",activation=None, use_bias=True, padding="same")(input1)
    x_1 = BatchNormalization()(x_1)
    x_2 = Conv2D(dim, kernel_size=1, strides=(1,1),kernel_initializer="he_normal",activation=None, use_bias=True, padding="same")(input2)
    x_2 = BatchNormalization()(x_2)
    x_3 = Add()([x_1,x_2])
    x_3 = Activation('relu')(x_3)
    x_3 = Conv2D(1, kernel_size=1, strides=(1,1),kernel_initializer="he_normal",activation=None, use_bias=True, padding="same")(x_3)
    x_3 = BatchNormalization()(x_3)
    x_3 = Activation('sigmoid')(x_3)
    x_4 = x_3
    for i in range(dim-1):
        x_4 = concatenate([x_4, x_3], axis = 3)
    x_5 = Multiply()([input1,x_4])

    return x_5

def Linknet_attention(pretrained_weights=None,
            input_size=(None, None, 3),
            n_filter=16,
            activation='relu',
            dropout=True, dropout_rate=0.5,
            batchnorm=True,
            loss=binary_crossentropy,
            optimizer=Adam(lr=1e-4)):
    """Build a standard UNet model.

    Arguments:
        pretrained_weights {str} -- path of the pretrained weights (default: {None})
        input_size {tuple} -- size of input images (default: {(None,None,3)})
        n_filter {int} -- number of filter of the first layer (default: {16})
        activation {str} -- activation function to use (default: {'relu'})
        dropout {bool} -- whether to use dropout layer (default: {True})
        dropout_rate {float} -- dropout rate (default: {0.5})
        batchnorm {bool} -- whether to use batch normalization layer (default: {True})
        loss {keras.losses} -- loss function to use (default: {binary_crossentropy})
        optimizer {keras.optimizers} -- optimizer to use (default: {Adam(lr=1e-4)})

    Returns:
        keras.models -- UNet model
    """

    # 3
    inputs = Input(input_size)

    # down path
    # n_filter
    conv1 = Conv2d_Block(inputs, 64, kernel_size=7, strides=(2, 2))
    # pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2,2))(conv1)

    # encode
    encode1 = Encoder(conv1, 64)
    encode2 = Encoder(encode1, 128)
    encode3 = Encoder(encode2, 256)
    # encode4=Encoder(encode3, 512)

    # decode
    # decode1=Decoder(encode4, 512,256)
    # decode1=decode1+encode4
    decode2 = Decoder(encode3, 256, 128)
    encode2 = AG(encode2,decode2,128)
    decode2 = add([decode2, encode2])
    decode3 = Decoder(decode2, 128, 64)
    encode1 = AG(encode1,decode3,64)
    decode3 = add([decode3, encode1])
    decode4 = Decoder(decode3, 64, 64)
    # decode4=add([decode4,encode1])

    fullconv1 = Conv2dT_Block(decode4, 32, kernel_size=2, strides=(2, 2), padding='same')
    conv2 = Conv2d_Block(fullconv1, 32, kernel_size=3, strides=(1, 1))
    fullconv2 = Conv2dT_Block(conv2, 1, kernel_size=1, strides=(1, 1), padding='same', activation='sigmoid')

    model = Model(inputs=inputs, outputs=fullconv2)

    model.compile(optimizer=optimizer, loss=loss, metrics=[f1, 'accuracy'])

    if (pretrained_weights):
        model.load_weights(filepath=pretrained_weights)

    return model



