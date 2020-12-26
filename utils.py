import numpy as np
import os
import struct
from optimizers import *

# some utility functions.

def shuffle_data(ori_data, eye=True, length=10000): # given the data(including labels), shuffle them and return.
    np.random.shuffle(ori_data)
    label = ori_data[:length,-1].astype(int)
    data = ori_data[:length,:-1]/255
    
    if eye:
        label=np.eye(10)[label]
    return data, label

def load_mnist(path, kind='train'): # load the mnist data easy and translate them to numpy arrays.
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,'%s-labels.idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images.idx3-ubyte'% kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

def test(net, ori_data): # test the trained net on the test set(or training set as well).
    data,label=shuffle_data(ori_data, eye=False)
    pred = np.argmax(net(data),1)
    acc = np.mean(pred==label)
    return acc

# optimizer names and settings for use.
opt_name = ['SGD', 'Momentum', 'Nesterov', 'Adagrad', 'RMSprop', 'Adam']
opt_ls = [SGD(1e-2, 0.9999), Momentum(1e-3, 0.9), Nesterov(1e-3, 0.9),
            AdaGrad(1e-1, 1e-9), RMSprop(1e-2, 1e-8, 0.9), Adam(1e-2, (0.9, 0.999), 1e-8)]