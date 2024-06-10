import os

#数据集路径
ROOT2 = "./dataset2/"
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

output_ch = 16
img1_ch =30
img2_ch =4 # 16
batch_size = 512
num_epochs = 3000
lr = 0.0001
num_epochs_decay = 10