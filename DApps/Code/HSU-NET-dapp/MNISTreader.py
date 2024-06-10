import os
import struct
import numpy as np
import torch

import cfg

def load_mnist(path = cfg.MNIST_ROOT,mode='train'):
    """Load MNIST data from `path`"""
    kind='train'
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 28,28)

    L = len(labels)
    if mode == "train":
        images=images[:int(L*0.8)]
        labels=labels[:int(L*0.8)]
    elif mode == "test":
        images = images[int(L*0.8):int(L*0.9)]
        labels = labels[int(L*0.8):int(L*0.9)]
    else:
        images = images[int(L*0.9):-1]
        labels = labels[int(L*0.9):-1]
    return images, labels