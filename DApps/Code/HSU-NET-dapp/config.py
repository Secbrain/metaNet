import os


#MNIST数据集路径
MNIST_ROOT = "../MNIST/"

#MNIST训练集图片路径
MNIST_images = os.path.join(MNIST_ROOT,"train-images.idx3-ubyte")
#MNIST训练集标签路径
MNIST_labels = os.path.join(MNIST_ROOT,"train-labels.idx1-ubyte")

#数据集路径
ROOT = "../dataset/"

Class_Values = {"Adware":0,"Benign":1,"Ransomware":2,"Scareware":3,"SMSMalware":4}

#网络参数
batch_size = 2
epoch = 2