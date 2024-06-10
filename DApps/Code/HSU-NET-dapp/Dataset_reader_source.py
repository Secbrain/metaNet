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
    img = []
    labels = []
    path = ""
    if mode == "train":
        path = train_txt_path
    elif mode == "test":
        path = test_txt_path
    else:
        path = valid_txt_path

    with open(path,"r") as f:
        for line in f.readlines():
            data_path,data_label = line.split()
            data = np.load(data_path)
            for j in range(10):
                dataj = tensor_split(torch.tensor(data[j]),2,[56,56])
                img.append(np.array([dataj]))
                labels.append(cfg.Class_Values[data_label])
            #data = tensor_split(torch.tensor(data), 3, [cfg.C, cfg.L, cfg.W])
            #img.append(data)
            #labels.append(cfg.Class_Values[data_label])
    return img,labels
