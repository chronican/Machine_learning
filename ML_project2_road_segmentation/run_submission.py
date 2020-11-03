# -*- coding: utf-8 -*-
"""
"""
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import os
import tensorflow as tf
tf.set_random_seed(1)

from keras.models import load_model

from data import testGenerator, save_result
from losses import dice_loss, bce_dice_loss
from metrics import f1
from mask_to_submission import make_submission



TEST_SIZE = 50
test_path = os.path.join("data", "test_set_images")

predict_path = "predict_images"
submission_path = "submission"
weight_path = "weights"

w = "weights_dlinknet_best.h5"



print("Load models and predict...")

results = 0

model=load_model(os.path.join(weight_path, w), custom_objects={"dice_loss": dice_loss, "f1": f1})
print('load model done')
testGene = testGenerator(test_path)
print('testGene done')
results += model.predict_generator(testGene, TEST_SIZE, verbose=1)
print('predict done')
save_result(predict_path, results)


print("Make submission...")
make_submission(predict_path, test_size=TEST_SIZE, submission_filename=os.path.join(submission_path, "submission_best.csv"))

print("Done!")

