import tensorflow as tf
slim = tf.contrib.slim

def get_placeholder_X(img_h, img_w, n_features):
    with tf.variable_scope("Placeholders_X"):
        X = tf.placeholder(dtype = tf.float32, shape = [None, img_h, img_w, n_features])
    return X
def get_placeholder_Y(n_labels):
    with tf.variable_scope("Placeholders_Y"):
        Y = tf.placeholder(dtype = tf.float32, shape = [None, n_labels])
    return Y

def forward(X, is_training = True):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn = tf.nn.relu,
                        normalizer_fn = slim.batch_norm,
                        normalizer_params = {'is_training' : is_training, 
                                             'updates_collections': tf.GraphKeys.UPDATE_OPS, 
                                             'decay' : 0.95}):
        with tf.variable_scope("Networks"):
            outputs = slim.conv2d(X, num_outputs = 8, kernel_size = [3, 3], stride = 1, padding = 'VALID') 
            outputs = slim.conv2d(outputs, num_outputs = 8, kernel_size = [3, 3], stride = 1, padding = 'VALID')
            outputs = slim.max_pool2d(outputs, [2, 2])
            outputs = slim.conv2d(outputs, num_outputs = 16, kernel_size = [3, 3], stride = 1, padding = 'VALID') 
            outputs = slim.conv2d(outputs, num_outputs = 16, kernel_size = [3, 3], stride = 1, padding = 'VALID')
            outputs = slim.max_pool2d(outputs, [2, 2])
            outputs = slim.flatten(outputs)
            outputs = slim.fully_connected(outputs, 512)
            outputs = slim.fully_connected(outputs, 128)
    with tf.variable_scope("Predictions"):
        outputs = slim.fully_connected(outputs, 10, activation_fn = None)
        predictions = tf.nn.softmax(outputs)
    return outputs, predictions

def train(LEARNING_RATE, outputs, Y):
    with tf.variable_scope("Trainings"):
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits = outputs))
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies([tf.group(*update_ops)]):
            train_op = optimizer.minimize(loss)
    return loss, train_op

def accuracy(predictions, Y):        
    with tf.variable_scope("Accuracys"):
        correct_prediction  = tf.equal(tf.argmax(Y, 1), tf.argmax(predictions, 1))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return correct_prediction, acc

def save():
    with tf.variable_scope("Savers"):
        saver = tf.train.Saver()
    return saver


        
