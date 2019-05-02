import tensorflow as tf
import numpy as np
from core import common
import network
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--TRAIN_DIR", help="Location of the trained model", default = "tmp/")
args = parser.parse_args()

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data() 

img_h = x_test.shape[1]
img_w = x_test.shape[2]

img_cahnnels = 1

x_test = x_test.reshape(-1, img_h, img_w, img_cahnnels).astype(np.float32)
x_test/=255.

n_classes = 10

y_test = common.convert_one_hot(y_test, n_classes)

n_features = x_test.shape[-1]
n_labels = y_test.shape[-1]

X = network.get_placeholder_X(img_h, img_w, n_features)
Y = network.get_placeholder_Y(n_labels)
outputs, predictions = network.forward(X, is_training = False)
correct_prediction, acc = network.accuracy(predictions, Y)

print("x_test.shape = ", x_test.shape)
print("y_test.shape = ", y_test.shape)

saver = tf.train.Saver()

with tf.Session() as sess:
    # from train_dir search the latest model
    ckpt = tf.train.get_checkpoint_state(args.TRAIN_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        # load trained model
        saver.restore(sess, ckpt.model_checkpoint_path)
        # get trained epoch num from checkpoint
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        accuracy_score = sess.run(acc, feed_dict = {X:x_test, Y:y_test})
        print("After %s training steps, test accuracy = %f" %(global_step, accuracy_score))
    else:
        print("No checkpoint file found")
