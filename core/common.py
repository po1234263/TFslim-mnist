import numpy as np

def convert_one_hot(array, n):
    array = np.eye(n, dtype = np.float32)[array]
    return array

def get_batches(X, Y, batch_sizes, shuffle_array_slice):
    batch_X = X[shuffle_array_slice]
    batch_Y = Y[shuffle_array_slice]
    return batch_X, batch_Y