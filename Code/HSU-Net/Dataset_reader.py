import glob
import torch
import numpy as np
import cfg
import os
root_path = cfg.ROOT
train_txt_path = os.path.join(root_path,"train.txt")
test_txt_path = os.path.join(root_path,"test.txt")
valid_txt_path = os.path.join(root_path,"valid.txt")

def tensor_split(t,dim,split_list):
    # t - Tensor need to be split
    # dim - dimension of Tensor t
    # split_list - dimension size list Tensor t will be splited to
    for d in range(dim):
        t = t.split(split_list[d],dim = d)[0]
        #np.split(t,split_list[d],axis=d)
    return t.numpy()

def get_dataset(mode="train"):
    img1 = []
    img2 = []
    labels = []
    path = ""
    if mode == "train":
        path = train_txt_path
    elif mode == "test":
        path = test_txt_path
    else:
        path = valid_txt_path

    with open(path,"r") as f:
        n = 0
        for line in f.readlines():
            data1_path,data2_path,data_label = line.split()
            data1 = np.load(data1_path)
            data2 = np.load(data2_path,allow_pickle=True).astype(float)
            data2 = data2.reshape(5,28,28)

            data1 = tensor_split(torch.tensor(data1), 3, [cfg.C, cfg.L, cfg.W])
            img1.append(data1)
            img2.append(data2)
            labels.append(cfg.Class_Values[data_label])

            #data = tensor_split(torch.tensor(data), 3, [cfg.C, cfg.L, cfg.W])
            #img.append(data)
            #labels.append(cfg.Class_Values[data_label])
    return img1,img2,labels
