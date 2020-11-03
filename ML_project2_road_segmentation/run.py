 #!/usr/bin/env python3

import numpy as np
np.random.seed(1)
import random
random.seed(1)
import os
import tensorflow as tf
tf.set_random_seed(1)

from keras.models import load_model

from data import testGenerator, save_result
from losses import dice_loss
from metrics import f1
from mask_to_submission import make_submission



TEST_SIZE = 50
test_path = os.path.join("data", "test_set_images")

predict_path = "predict_images"
submission_path = "submission"
weight_path = "weights"
weight_list=["weights_segnet.h5", "weights_unet.h5", "weights_dunet.h5","weights_unet_attention.h5"
             ,"weights_linknet.h5","weights_dlinknet.h5",'weights_linknet_attention.h5']


method=int(input("choose method:1.validate single modle 2.calculate the voting"))
vote_result=0
if method==1:
    num=int(input("choose  model number you want to validate: 1.segnet 2.unet 3. dunet 4.unet_attention 5.linknet 6.dlinknet 7.link_attention"))
    w=weight_list[num-1]
    model = load_model(os.path.join(weight_path, w), custom_objects={"dice_loss": dice_loss, "f1": f1})
    test = testGenerator(test_path)
    predict_result= model.predict_generator(test, TEST_SIZE, verbose=1)
    print(type(predict_result))
    save_result(predict_path, predict_result)
    
if method==2:
    print("calculate the voting of different models")
    for w in weight_list:
        model= load_model(os.path.join(weight_path, w), custom_objects={"dice_loss": dice_loss, "f1": f1})
        test=testGenerator(test_path)
        vote_result+=model.predict_generator(test, TEST_SIZE, verbose=1)
        save_result(predict_path,vote_result)

print("make submission")
make_submission(predict_path, test_size=TEST_SIZE, submission_filename=os.path.join(submission_path, "submission.csv"))


