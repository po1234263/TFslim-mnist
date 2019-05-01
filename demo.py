import tensorflow as tf
import numpy as np
from core import common
slim = tf.contrib.slim

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

img_h = x_train.shape[1]
img_w = x_train.shape[2]

x_train = x_train.reshape(-1, img_h, img_w, 1).astype(np.float32)
x_test = x_test.reshape(-1, img_h, img_w, 1).astype(np.float32)

x_train/=255.
x_test/=255.

y_train = common.convert_one_hot(y_train, 10)
y_test = common.convert_one_hot(y_test, 10)

x_val = x_train[-3000:]
y_val = y_train[-3000:]
x_train = x_train[:-3000]
y_train = y_train[:-3000]

n_examples = x_train.shape[0]
n_features = x_train.shape[-1]
n_labels = y_train.shape[-1]

print("x_train.shape = ", x_train.shape)
print("y_train.shape = ", y_train.shape)
print("x_val.shape = ", x_val.shape)
print("y_val.shape = ", y_val.shape)
print("x_test.shape = ", x_test.shape)
print("y_test.shape = ", y_test.shape)

with tf.variable_scope("Placeholders"):
    X = tf.placeholder(dtype = tf.float32, shape = [None, img_h, img_w, n_features])
    Y = tf.placeholder(dtype = tf.float32, shape = [None, n_labels])

with tf.variable_scope("Networks"):
    outputs = slim.conv2d(X, num_outputs = 8, kernel_size = [3, 3], stride = 1, activation_fn = tf.nn.relu, padding = 'VALID') 
    outputs = slim.conv2d(outputs, num_outputs = 8, kernel_size = [3, 3], stride = 1, activation_fn = tf.nn.relu, padding = 'VALID')
    outputs = slim.max_pool2d(outputs, [2, 2])
    outputs = slim.conv2d(outputs, num_outputs = 16, kernel_size = [3, 3], stride = 1, activation_fn = tf.nn.relu, padding = 'VALID') 
    outputs = slim.conv2d(outputs, num_outputs = 16, kernel_size = [3, 3], stride = 1, activation_fn = tf.nn.relu, padding = 'VALID')
    outputs = slim.max_pool2d(outputs, [2, 2])
    outputs = slim.flatten(outputs)
    outputs = slim.fully_connected(outputs, 256, activation_fn = tf.nn.relu)
    outputs = slim.fully_connected(outputs, 128, activation_fn = tf.nn.relu)
    outputs = slim.fully_connected(outputs, 10, activation_fn = None)
    predictions = tf.nn.softmax(outputs)

with tf.variable_scope("Trainings"):
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits = outputs))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    correct_prediction  = tf.equal(tf.argmax(Y, 1), tf.argmax(predictions, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

TRAIN_EPOCHS = 10
BATCH_SIZES = 16
TOTAL_BATCHES = int(n_examples / BATCH_SIZES + 0.5)
DISPLAY_EPOCH = 1

print("\nStart Training!!!\n")

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
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
            _, batch_loss = sess.run([optimizer, loss], feed_dict = {X:batch_X, Y:batch_Y})
            avg_loss+=batch_loss
        avg_loss/=TOTAL_BATCHES
        if(epoch % DISPLAY_EPOCH == 0):
            val_acc = sess.run(acc, feed_dict = {X:x_val, Y:y_val})
            print("Epoch %d: Loss = %f, val_acc = %f" %(epoch + 1, avg_loss, val_acc))

    print("Testing set accuracy = %f" %(sess.run(acc, feed_dict = {X:x_test, Y:y_test})))
        
