import os
import random

import cfg
import numpy as np
import glob

root_path = cfg.ROOT
root2_path = cfg.ROOT2
'''
#大标签均分
def split_dataset():
    img_label = []
    label_list = os.listdir(root_path)
    for label in label_list:
        mini_path = os.path.join(root_path,label)
        mini_label_list = os.listdir(mini_path)
        label_data_list = []

        for mini_lable in mini_label_list:
            data_list = glob.glob(os.path.join(mini_path,mini_lable,"*.npy"))
            for data_path in data_list:
                data_path2 = (os.path.join(root2_path, label, mini_lable, data_path.split('/')[-1]))
                label_data_list.append((data_path,data_path2))
                #label_data_list.append(data_path)
                #label_data_list.append(data_path)
                #label_data_list.append(data_path)
        #random.shuffle(label_data_list)
        L = len(label_data_list)
        with open(os.path.join(root_path, "train.txt"), "a") as f:
            for i in range(int(L * 0.7)):
                f.write(label_data_list[i][0] + " " + label_data_list[i][1] + " " + label)
                f.write('\n')
        with open(os.path.join(root_path, "test.txt"), "a") as f:
            for i in range(int(L * 0.7), int(L * 0.9)):
                f.write(label_data_list[i][0] + " " + label_data_list[i][1] + " " + label)
                f.write('\n')
        with open(os.path.join(root_path, "valid.txt"), "a") as f:
            for i in range(int(L * 0.9), L):
                f.write(label_data_list[i][0] + " " + label_data_list[i][1] + " " + label)
                f.write('\n')
    print("output train.txt,test.txt,valid.txt in",root_path)
    '''
#小标签均分
def split_dataset():
    label_list = os.listdir(root_path)
    for label in label_list:
        if label == 'Benign':
            label_binary = 'Benign'
        else:
            label_binary = 'Malicious'
        mini_path = os.path.join(root_path,label)
        if not os.path.isdir(mini_path):
            continue
        mini_label_list = os.listdir(mini_path)
        for mini_lable in mini_label_list:
            label_data_list = []
            data_list = glob.glob(os.path.join(mini_path,mini_lable,"*.npy"))
            for data_path in data_list:
                data_path2 = (os.path.join(root2_path,label,mini_lable,data_path.split('/')[-1]))
                label_data_list.append((data_path, data_path2))
                # label_data_list.append((data_path, data_path2))
                # label_data_list.append((data_path, data_path2))
                # label_data_list.append((data_path, data_path2))
                #label_data_list.append(data_path)
                #label_data_list.append(data_path)
                #label_data_list.append(data_path)
            random.shuffle(label_data_list)
            L = len(label_data_list)
            with open(os.path.join(root_path,"train.txt"),"a") as f:
                for i in range(int(L*0.8)):
                    f.write(label_data_list[i][0] + " " + label_data_list[i][1] + " " + label_binary)
                    f.write('\n')
            with open(os.path.join(root_path,"test.txt"),"a") as f:
                # for i in range(int(L*0.7),int(L*0.9)):
                for i in range(int(L*0.8),L):
                    f.write(label_data_list[i][0] + " " + label_data_list[i][1] + " " + label_binary)
                    f.write('\n')
            with open(os.path.join(root_path,"valid.txt"),"a") as f:
                for i in range(int(L*0.8),L):
                # for i in range(int(L * 0.9), L):
                    f.write(label_data_list[i][0] + " " + label_data_list[i][1] + " " + label_binary)
                    f.write('\n')
    print("output train.txt,test.txt,valid.txt in",root_path)

def split_dataset_bigclass():
    label_list = os.listdir(root_path)
    for label in label_list:
        if label == 'Benign':
            label_binary = 'Benign'
        else:
            label_binary = 'Malicious'
        mini_path = os.path.join(root_path,label)
        if not os.path.isdir(mini_path):
            continue
        mini_label_list = os.listdir(mini_path)
        label_data_list = []
        for mini_lable in mini_label_list:
            data_list = glob.glob(os.path.join(mini_path,mini_lable,"*.npy"))
            for data_path in data_list:
                data_path2 = (os.path.join(root2_path,label,mini_lable,data_path.split('/')[-1]))
                label_data_list.append((data_path, data_path2))
                # label_data_list.append((data_path, data_path2))
                # label_data_list.append((data_path, data_path2))
                # label_data_list.append((data_path, data_path2))
                #label_data_list.append(data_path)
                #label_data_list.append(data_path)
                #label_data_list.append(data_path)
        random.shuffle(label_data_list)
        L = len(label_data_list)
        with open(os.path.join(root_path,"train.txt"),"a") as f:
            for i in range(int(L*0.8)):
                f.write(label_data_list[i][0] + " " + label_data_list[i][1] + " " + label_binary)
                f.write('\n')
        with open(os.path.join(root_path,"test.txt"),"a") as f:
            # for i in range(int(L*0.7),int(L*0.9)):
            for i in range(int(L*0.8),L):
                f.write(label_data_list[i][0] + " " + label_data_list[i][1] + " " + label_binary)
                f.write('\n')
        with open(os.path.join(root_path,"valid.txt"),"a") as f:
            for i in range(int(L*0.8),L):
            # for i in range(int(L * 0.9), L):
                f.write(label_data_list[i][0] + " " + label_data_list[i][1] + " " + label_binary)
                f.write('\n')
    print("output train.txt,test.txt,valid.txt in",root_path)

# split_dataset()
split_dataset_bigclass()