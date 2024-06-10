import os
import random

import cfg
import numpy as np
import glob

def split_dataset():
    img_label = []
    root_path = cfg.ROOT
    label_list = os.listdir(root_path)
    for label in label_list:
        mini_path = os.path.join(root_path,label)
        mini_label_list = os.listdir(mini_path)
        label_data_list = []
        for mini_lable in mini_label_list:
            data_list = glob.glob(os.path.join(mini_path,mini_lable,"*.npy"))
            for data_path in data_list:
                label_data_list.append(data_path)
            random.shuffle(label_data_list)
            with open(os.path.join(root_path,"train.txt"),"a") as f:
                for i in label_data_list[:int(len(label_data_list)*0.7)]:
                    f.write(i+" "+label)
                    f.write('\n')
            with open(os.path.join(root_path,"test.txt"),"a") as f:
                for i in label_data_list[int(len(label_data_list)*0.7):int(len(label_data_list)*0.9)]:
                    f.write(i+" "+label)
                    f.write('\n')
            with open(os.path.join(root_path,"valid.txt"),"a") as f:
                for i in label_data_list[int(len(label_data_list)*0.9):]:
                    f.write(i+" "+label)
                    f.write('\n')
    print("output train.txt,test.txt,valid.txt in",root_path)
split_dataset()