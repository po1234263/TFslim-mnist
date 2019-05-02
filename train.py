import tensorflow as tf
import numpy as np
from core import common
import network
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--VAL_SIZE", help = "Validation set size", default = 5000)
parser.add_argument("--TRAIN_EPOCHS", help="Total train epoch", default = 30)
parser.add_argument("--BATCH_SIZES", help="Number of a batch", default = 32)
parser.add_argument("--LEARNING_RATE", help="learning rate", default = 0.01)
parser.add_argument("--DISPLAY_EPOCH", help="How often does it show training loss and validation accuracy", default = 1)
parser.add_argument("--TRAIN_DIR", help="Location of the trained model", default = "tmp/")
parser.add_argument("--LOG_DIR", help="Location of the logs(for tensorboard use)", default = "logs/")
args = parser.parse_args()

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

img_h = x_train.shape[1]
img_w = x_train.shape[2]

img_cahnnels = 1

x_train = x_train.reshape(-1, img_h, img_w, img_cahnnels).astype(np.float32)

x_train/=255.

n_classes = 10

y_train = common.convert_one_hot(y_train, n_classes)

val_set_size = args.VAL_SIZE

x_val = x_train[-val_set_size:]
y_val = y_train[-val_set_size:]
x_train = x_train[:-val_set_size]
y_train = y_train[:-val_set_size]

n_examples = x_train.shape[0]
n_features = x_train.shape[-1]
n_labels = y_train.shape[-1]

print("x_train.shape = ", x_train.shape)
print("y_train.shape = ", y_train.shape)
print("x_val.shape = ", x_val.shape)
print("y_val.shape = ", y_val.shape)


TRAIN_EPOCHS = args.TRAIN_EPOCHS
BATCH_SIZES = args.BATCH_SIZES
LEARNING_RATE = args.LEARNING_RATE
TOTAL_BATCHES = int(n_examples / BATCH_SIZES + 0.5)
DISPLAY_EPOCH = args.DISPLAY_EPOCH


X = network.get_placeholder_X(img_h, img_w, n_features)
Y = network.get_placeholder_Y(n_labels)
outputs, predictions = network.forward(X, is_training = True)
loss, train_op = network.train(LEARNING_RATE, outputs, Y)
correct_prediction, acc = network.accuracy(predictions, Y)
saver = network.save()

logs_path = args.LOG_DIR
tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", acc)
merged_summary_op = tf.summary.merge_all()

print("\nStart Training!!!\n")

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    summary_writer = tf.summary.FileWriter(logs_path, graph = tf.get_default_graph())
    val_writer = tf.summary.FileWriter(logs_path + 'val', graph = tf.get_default_graph())
    for epoch in range(TRAIN_EPOCHS):
        shuffle_array = np.arange(n_examples)
        np.random.shuffle(shuffle_array)
        avg_loss = 0
        for batch in range(TOTAL_BATCHES):
            batch_start = batch * BATCH_SIZES
            batch_end = batch_start + BATCH_SIZES

            if(batch_end > n_examples):
                n_out_array = n_examples - batch_end
                new_shuffle_array = np.arange(n_examples)
                np.random.shuffle(new_shuffle_array)
                shuffle_array = np.concatenate((shuffle_array, new_shuffle_array[:n_out_array]))

            batch_X, batch_Y = common.get_batches(x_train, y_train, BATCH_SIZES, shuffle_array[batch_start : batch_end])
            _, batch_loss, summary = sess.run([train_op, loss, merged_summary_op], feed_dict = {X:batch_X, Y:batch_Y})
            summary_writer.add_summary(summary, epoch * TOTAL_BATCHES + batch)
            avg_loss+=batch_loss
        avg_loss/=TOTAL_BATCHES
        if(epoch % DISPLAY_EPOCH == 0):
            val_acc, summary = sess.run([acc, merged_summary_op], feed_dict = {X:x_val, Y:y_val})
            val_writer.add_summary(summary, epoch + 1)
            print("Epoch %3d: Loss = %f, val_acc = %3.5f" %(epoch + 1, avg_loss, val_acc))
    saver.save(sess, args.TRAIN_DIR + 'model.ckpt', global_step = epoch + 1)