#!/usr/bin/env python3

import numpy as np
np.random.seed(1)
import random
random.seed(1)
import tensorflow as tf
tf.set_random_seed(1)

import os
import skimage.io as io
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import cv2
def make_dir(path):
    """Make directory."""
    if not os.path.exists(path):
        os.makedirs(path)


def build_train_val(train_path,val_path,val_size=0.2,seed=1):
    '''Build training and validation data set
    
    Arguments:
        train_path{str}: the path that saving training images
        val_path{str}: the path that saving validation images
    Keyword Arguments:
        val_size{double}: ratio of validation images(default: 0.2)
        seed{int}: random seed(default :1)
            
    '''


    for i in range(1,101):
        im=cv2.imread(os.path.join(train_path, 'images','satImage_%.3d.png' % i))
        #im = cv2.imread(os.path.join(train_path, 'images', i))
        ma = cv2.imread(os.path.join(train_path, 'groundtruth', 'satImage_%.3d.png' % i))
        imf = cv2.flip(im, 1)
        cv2.imwrite(os.path.join(train_path, 'images', 'satImage_%.3d_f.png' % i),imf)
        maf = cv2.flip(ma, 1)
        cv2.imwrite(os.path.join(train_path,'groundtruth' , 'satImage_%.3d_f.png' % i),maf)

        im90 = np.rot90(im)
        cv2.imwrite(os.path.join(train_path, 'images', 'satImage_%.3d_90.png' % i), im90)
        im180 = np.rot90(im90)
        cv2.imwrite(os.path.join(train_path, 'images', 'satImage_%.3d_180.png' % i), im180)
        im270 = np.rot90(im180)
        cv2.imwrite(os.path.join(train_path, 'images', 'satImage_%.3d_270.png' % i), im270)
        imf90 = np.rot90(imf)
        cv2.imwrite(os.path.join(train_path, 'images', 'satImage_%.3d_f_90.png' % i), imf90)
        imf180 = np.rot90(imf90)
        cv2.imwrite(os.path.join(train_path, 'images', 'satImage_%.3d_f_180.png' % i), imf180)
        imf270 = np.rot90(imf180)
        cv2.imwrite(os.path.join(train_path, 'images', 'satImage_%.3d_f_270.png' % i), imf270)

        ma90 = np.rot90(ma)
        cv2.imwrite(os.path.join(train_path, 'groundtruth', 'satImage_%.3d_90.png' % i), ma90)
        ma180 = np.rot90(ma90)
        cv2.imwrite(os.path.join(train_path, 'groundtruth', 'satImage_%.3d_180.png' % i), ma180)
        ma270 = np.rot90(ma180)
        cv2.imwrite(os.path.join(train_path, 'groundtruth', 'satImage_%.3d_270.png' % i), ma270)
        maf90 = np.rot90(maf)
        cv2.imwrite(os.path.join(train_path, 'groundtruth', 'satImage_%.3d_f_90.png' % i), maf90)
        maf180 = np.rot90(maf90)
        cv2.imwrite(os.path.join(train_path, 'groundtruth', 'satImage_%.3d_f_180.png' % i), maf180)
        maf270 = np.rot90(maf180)
        cv2.imwrite(os.path.join(train_path, 'groundtruth', 'satImage_%.3d_f_270.png' % i), maf270)
    
    train_val_images = os.listdir(os.path.join(train_path, 'images'))
    print(train_val_images)

    # Split image into train and validation set
    train_images, val_images = train_test_split(train_val_images, test_size=val_size, random_state=seed)

    # Build new folders for validation set
    make_dir(val_path)
    make_dir(os.path.join(val_path, 'images'))
    make_dir(os.path.join(val_path, 'groundtruth'))
    
    # Move validation images to new folders
    for im in val_images:
        os.rename(os.path.join(train_path, 'images', im), os.path.join(val_path, 'images', im))
        os.rename(os.path.join(train_path, 'groundtruth', im), os.path.join(val_path, 'groundtruth', im))
    

def preprocess_mask(mask):
    """Preprocessing function for masks."""
    mask = mask/255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return mask

  
def preprocess_img(img):
    """Preprocessing function for images."""
    return img/255

def trainvalGenerator(batch_size, 
                      train_path, val_path,
                      image_folder = 'images', mask_folder = 'groundtruth',
                      target_size = (400,400), seed = 1):  
    
    """Generator for training and validaton set.
    
    Arguments:
        batch_size {int}: number of images in each batch
        train_path {str}: path of the training set
        val_path {str}: path of the validation set
        image_folder {str}: image folder's name (default: {"images"})
        mask_folder {str}: mask folder's name (default: {"groundtruth"})
        target_size {tuple}: size of targe images (default: {(400,400)})
        seed {int}: random seed (default: {1})
    
    Returns:
        (trainGen, valGen) -- tuple of generators for training and validation set
    """ 

    # generating training images ans masks
    image_data_generator = ImageDataGenerator(
                     rotation_range=45,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='reflect',
                     preprocessing_function= preprocess_img
                     )
    
    mask_data_generator = ImageDataGenerator(
                     rotation_range=45,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='reflect',
                     preprocessing_function= preprocess_img
                     )
    
    
    
    image_generator_train = image_data_generator.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = "rgb",
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = train_dir,
        save_prefix  = "image",
        seed = seed)
    
    
   
    
    
    mask_generator_train = mask_data_generator.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = "grayscale",
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = train_dir,
        save_prefix  = "mask",
        seed = seed)
    
    #generating Validation images and masks


    image_data_generator = ImageDataGenerator(preprocessing_function=preprocess_img)
    mask_data_generator = ImageDataGenerator(preprocessing_function=preprocess_mask)
    
    image_generator_val = image_data_generator.flow_from_directory(
        val_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = "rgb",
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = val_dir,
        save_prefix  = "image",
        seed = seed+1,
        shuffle = False
    )
    
    mask_generator_val = mask_data_generator.flow_from_directory(
        val_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = "grayscale",
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = val_dir,
        save_prefix  = "mask",
        seed = seed+1,
        shuffle = False
    )
    
    return zip(image_generator_train, mask_generator_train), zip(image_generator_val, mask_generator_val)


def testGenerator(test_path, num_image = 50):
    """Generator for test set.
    
    Arguments:
        test_path {set} -- path of the test set
        num_image {int} -- number of images in the test set (default: {50})
    """
    for i in range(1, num_image+1):
        img = io.imread(os.path.join(test_path, "test_%d"%i, "test_%d.png"%i))
        img = img / 255
        img = np.reshape(img,(1,)+img.shape)
        yield img



def save_result(save_path, npyfile):
    """Save predicted images to path.

    Arguments:
        save_path {str} -- path of the folder to save 
        npyfile {numpy.ndarray} -- numpy array of the predict images
    """
    make_dir(save_path)
    for i, item in enumerate(npyfile):
        img = item[:,:,0]
        io.imsave(os.path.join(save_path, '%.3d.png'%(i+1)), img)
