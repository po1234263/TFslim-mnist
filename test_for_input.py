import tensorflow as tf
import numpy as np
from core import common
import network
import argparse
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--TRAIN_DIR", help="Location of the trained model", default = "tmp/")
parser.add_argument("--TEST_IMG", help="Location of the trained model", default = "tmp/")
args = parser.parse_args()

def simple_test(test_img):
    if(len(test_img) == 0):
        x_test = cv2.imread(args.TEST_IMG, cv2.IMREAD_GRAYSCALE)
    else:
        x_test = test_img
    img_h = x_test.shape[0]
    img_w = x_test.shape[1]

    if(img_h != 28 or img_w != 28):
        x_test = cv2.resize(x_test, (28, 28), interpolation = cv2.INTER_NEAREST)

    img_cahnnels = 1

    x_test = x_test.reshape(1, img_h, img_w, img_cahnnels).astype(np.float32)
    x_test/=255.

    n_features = x_test.shape[-1]

    X = network.get_placeholder_X(img_h, img_w, n_features)

    _, predictions = network.forward(X, is_training = False)

    print("x_test.shape = ", x_test.shape)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # from train_dir search the latest model
        ckpt = tf.train.get_checkpoint_state(args.TRAIN_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            # load trained model
            saver.restore(sess, ckpt.model_checkpoint_path)
            predict_value = sess.run(predictions, feed_dict = {X:x_test})
            print("The predicted value of the input image is %d" %(np.argmax(predict_value)))
        else:
            print("No checkpoint file found")

if __name__ == '__main__':
    simple_test(test_img = [])