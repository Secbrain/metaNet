import os

ROOT = "../dataset/"
ROOT2 = "../dataset2/"
trainbl = 0.8
model_save_path = "./models/"
result_path = "./result/"


#Data size
C = 1
L = 56
W = 56

Class_Values = {"Adware":0,"Benign":1,"Ransomware":2,"Scareware":3,"SMSMalware":4}

#Network parameters

output_ch = 5
img1_ch =1
img2_ch =5
batch_size = 20
num_epochs = 2
lr = 0.0001
num_epochs_decay = 10