import os

#MNIST数据集路径
MNIST_ROOT = "../MNIST/"
#MNIST训练集图片路径
MNIST_images = os.path.join(MNIST_ROOT,"train-images.idx3-ubyte")
#MNIST训练集标签路径
MNIST_labels = os.path.join(MNIST_ROOT,"train-labels.idx1-ubyte")

#数据集路径
ROOT = "./dataset1/"
ROOT2 = "./dataset2/"
# ROOT = "/mnt/traffic/luohao/data/mnist/MNIST/raw"
#训练集占比
trainbl = 0.8
#模型保存路径
model_save_path = "./models/"
#测试结果生成路径
result_path = "./result/"


#数据size
C = 1
L = 56
W = 56

Class_Values = {"Adware":0,"Benign":1,"Ransomware":2,"Scareware":3,"SMSMalware":4}
# Class_Values = {"Benign":0,"Malicious":1}

#网络参数

output_ch = 5
img1_ch =10
img2_ch =5
batch_size = 20
num_epochs = 2
lr = 0.0001
num_epochs_decay = 10