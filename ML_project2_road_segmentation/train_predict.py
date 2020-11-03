#!/usr/bin/env python3

import numpy as np
np.random.seed(1)
import random
random.seed(1)
import os
import tensorflow as tf
tf.set_random_seed(1)

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger

#from data import build_train_val, trainvalGenerator, testGenerator, save_result#, splite_image
from data_old import build_train_val, trainvalGenerator, testGenerator, save_result
from losses import dice_loss
from mask_to_submission import make_submission

from model_segnet import segnet

from model_unet import  unet
from model_dunet import unet_dilated
from model_unet_attention  import unet_Attention

from model_Dlinknet import DLinknet
from model_linknet import Linknet
from model_attention_linknet import Linknet_attention


from metrics import f1

NUM_EPOCH = 50 #the number of epochs, which maybe different for different 
NUM_TRAINING_STEP =  1000 
NUM_VALIDATION_STEP = 80
TEST_SIZE = 50

# paths
train_path = os.path.join("data", "training")
val_path = os.path.join("data", "validation")
test_path = os.path.join("data", "test_set_images")

predict_path = "predict_images" 
submission_path = "submission"
weight_path = "weights"
csv_path = "csv" # path to save the the csvlog file

if not os.path.exists(val_path):
    print("Validation set have not been built.")
    print("Build validation set now")
    build_train_val(train_path, val_path, val_size=0.2)
else:
    print("Have found training and validation data set...")





# Build generator for training and validation set
trainGen, valGen = trainvalGenerator(batch_size=2, 
                                     train_path=train_path, val_path=val_path,
                                     image_folder='images', mask_folder='groundtruth',
                                     target_size = (400, 400), seed = 1)



print("Build model and training...")



# Build models
print("Build the standard Seg-Net model ...")
model_segnet_1 = segnet(n_filter=32, activation='relu',  loss=dice_loss)

print("Build the standard U-Net model ...")
model_unet_1 = unet(n_filter=32, activation='relu', dropout=True, dropout_rate=0.2,  loss=dice_loss)

print("Build the U-Net model with dilated block...")
model_dunet_1 = unet_dilated(n_filter=32, activation='relu', dropout=True, dropout_rate=0.2, loss=dice_loss)

print("Build the U-Net model with attention gate...")
model_unet_Attention_1 = unet_Attention(n_filter=32,dropout=True, dropout_rate=0.2, activation='relu',  loss=dice_loss)

print("Build the standard Link-Net model...")
model_linknet_1 = Linknet(n_filter=32, activation='relu',  loss=dice_loss)

print("Build the Link-Net model with dilated block...")
model_dlinknet_1 = DLinknet(n_filter=64, activation='relu',  loss=dice_loss)

print("Build the Link-Net model with attention gate...")
model_Linknet_attention_1 = Linknet_attention(n_filter=64, activation='relu',  loss=dice_loss)

# Callback functions
callbacks_segnet = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=1e-4),
    ModelCheckpoint(os.path.join(weight_path, 'weights_segnet.h5'), monitor='val_loss', save_best_only=True, verbose=1),
    CSVLogger(os.path.join(csv_path, 'segnet.csv'), separator=',', append=False)
]

callbacks_unet = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=1e-4),
    ModelCheckpoint(os.path.join(weight_path, 'weights_unet.h5'), monitor='val_loss', save_best_only=True, verbose=1),
    CSVLogger(os.path.join(csv_path, 'unet.csv'), separator=',', append=False)
]

callbacks_dunet = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=1e-4),
    ModelCheckpoint(os.path.join(weight_path, 'weights_dunet.h5'), monitor='val_loss', save_best_only=True, verbose=1),
    CSVLogger(os.path.join(csv_path, 'dunet.csv'), separator=',', append=False)
]

callbacks_unet_attention = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=1e-4),
    ModelCheckpoint(os.path.join(weight_path, 'weights_unet_attention.h5'), monitor='val_loss', save_best_only=True, verbose=1),
    CSVLogger(os.path.join(csv_path, 'unet_attention.csv'), separator=',', append=False)
]

callbacks_linknet = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=1e-4),
    ModelCheckpoint(os.path.join(weight_path, 'weights_linknet.h5'), monitor='val_loss', save_best_only=True, verbose=1),
    CSVLogger(os.path.join(csv_path, 'linknet.csv'), separator=',', append=False)
]

callbacks_dlinknet = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=1e-4),
    ModelCheckpoint(os.path.join(weight_path, 'weights_dlinknet.h5'), monitor='val_loss', save_best_only=True, verbose=1),
    CSVLogger(os.path.join(csv_path, 'dlinknet.csv'), separator=',', append=False)
]

callbacks_linknet_attention = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=1e-4),
    ModelCheckpoint(os.path.join(weight_path, 'weights_linknet_attention.h5'), monitor='val_loss', save_best_only=True, verbose=1),
    CSVLogger(os.path.join(csv_path, 'linknet_attention.csv'), separator=',', append=False)
]






# Training



history_segnet = model_segnet_1.fit_generator(generator=trainGen, steps_per_epoch=NUM_TRAINING_STEP,
                                    validation_data=valGen, validation_steps=NUM_VALIDATION_STEP,
                                    epochs=NUM_EPOCH, callbacks=callbacks_segnet)

history_unet = model_unet_1.fit_generator(generator=trainGen, steps_per_epoch=NUM_TRAINING_STEP,
                                    validation_data=valGen, validation_steps=NUM_VALIDATION_STEP,
                                    epochs=NUM_EPOCH, callbacks=callbacks_unet)

history_dunet = model_dunet_1.fit_generator(generator=trainGen, steps_per_epoch=NUM_TRAINING_STEP,
                                    validation_data=valGen, validation_steps=NUM_VALIDATION_STEP,
                                    epochs=NUM_EPOCH, callbacks=callbacks_dunet)

history_unet_Attention = model_unet_Attention_1.fit_generator(generator=trainGen, steps_per_epoch=NUM_TRAINING_STEP,
                                    validation_data=valGen, validation_steps=NUM_VALIDATION_STEP,
                                    epochs=NUM_EPOCH, callbacks=callbacks_unet_attention)

history_linknet = model_linknet_1.fit_generator(generator=trainGen, steps_per_epoch=NUM_TRAINING_STEP,
                                    validation_data=valGen, validation_steps=NUM_VALIDATION_STEP,
                                    epochs=NUM_EPOCH, callbacks=callbacks_linknet)

history_dlinknet = model_dlinknet_1.fit_generator(generator=trainGen, steps_per_epoch=NUM_TRAINING_STEP,
                                    validation_data=valGen, validation_steps=NUM_VALIDATION_STEP,
                                    epochs=NUM_EPOCH, callbacks=callbacks_dlinknet)

history_linknet_attention = model_Linknet_attention_1.fit_generator(generator=trainGen, steps_per_epoch=NUM_TRAINING_STEP,
                                    validation_data=valGen, validation_steps=NUM_VALIDATION_STEP,
                                    epochs=NUM_EPOCH, callbacks=callbacks_linknet_attention)




