# -*- coding: utf-8 -*-
import os
from lxml import etree
from threading import *
from time import sleep
import json
import xlrd
from xlrd import xldate_as_tuple
import datetime
import numpy as np
import pandas as pd
import time
import csv
import io
from scapy.utils import PcapReader as PR
from scapy.all import rdpcap
from scapy.all import wrpcap
from scapy.all import PcapWriter
import sys
import scapy;
# from scapy_ssl_tls.ssl_tls import TLS;
import random
# import dataframe_image as dfi
# from PIL import Image
import cv2
import gc
import shutil
import matplotlib.mlab as mlab  
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def freqres_filtering(allfiles_dir, frequent_json_dir, filtering_output_dir, minlength, minnum):

    if not os.path.exists(filtering_output_dir):
        os.mkdir(filtering_output_dir)
    
    freqres_filtering = {}
    for json_name in os.listdir(frequent_json_dir):
        json_path = os.path.join(frequent_json_dir, json_name);
        if os.path.splitext(json_path)[1]=='.json':
            json_front = json_name.split('.')[0]
            
            json_val = None
            with open(json_path) as fp:
                json_val = json.load(fp)

            json_ok = []
            for val in json_val:
                if (len(val[0]) >= minlength) and (val[1] >= minnum):
                    json_ok.append(val)

            json_ok.sort(key=lambda x: (len(x[0]) + ((x[1]*1.0)/5)), reverse=True)

            data_json = json.dumps(json_ok, cls=NpEncoder)
            fileObject = open(os.path.join(filtering_output_dir, json_name), 'w')
            fileObject.write(data_json)
            fileObject.close()
            freqres_filtering[json_front] = json_ok
    
    data_json = json.dumps(freqres_filtering, cls=NpEncoder)
    fileObject = open(filtering_output_dir + '.json', 'w')
    fileObject.write(data_json)
    fileObject.close()
    print("freqres_filtering保存完毕！")

def unique_features(front_frequent_path):
    freqres_filtering_unique = {}

    json_val = None
    with open(front_frequent_path) as fp:
        json_val = json.load(fp)
    
    json_map = {}
    json_set = {}
    class_names = json_val.keys()
    for class_name in class_names:
        json_map_val = {}
        for val in json_val[class_name]:
            val_str = ','.join([str(val_val) for val_val in val[0]])
            json_map_val[val_str] = val
        json_map[class_name] = json_map_val
        json_set[class_name] = set(json_map_val.keys())

    for class_name in class_names:
        unique_list = []
        json_map_val = json_map[class_name]
        class_set = json_set[class_name]
        else_set = set()
        for else_class_name in class_names:
            if else_class_name != class_name:
                else_set = else_set | json_set[else_class_name]
        unique_set = class_set - else_set
        for unique_set_val in unique_set:
            unique_list.append(json_map_val[unique_set_val])
        unique_list.sort(key=lambda x: (len(x[0]) + ((x[1]*1.0)/5)), reverse=True)
        freqres_filtering_unique[class_name] = unique_list
    
    data_json = json.dumps(freqres_filtering_unique, cls=NpEncoder)
    fileObject = open(os.path.splitext(front_frequent_path)[0]+'-unique-class.json', 'w')
    fileObject.write(data_json)
    fileObject.close()

def meta_features(frequent_path, wide, front_frequent_number, output_dir):
    meta_features_unique = {}

    json_val = None
    with open(frequent_path) as fp:
        json_val = json.load(fp)
    
    csvdir = "andmal/andmal2020/mal-new-numlog2"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)

    for csv_name in os.listdir(csvdir):
        csv_path = os.path.join(csvdir, csv_name);
        if os.path.splitext(csv_path)[1]=='.csv':

            meta_val = np.zeros((front_frequent_number, wide), dtype=np.int32)
            csv_name_front = csv_name.split('.')[0]

            csvfile = io.open(csv_path,'r',encoding="utf-8")
            df = pd.read_csv(csvfile,header=None)
            csvfile.close()

            frequent_list = json_val[csv_name_front]
            test_words = []
            num_map = {}

            fre_lines = front_frequent_number
            if front_frequent_number > len(frequent_list):
                fre_lines = len(frequent_list)
            for line in range(len(df)):
                test_text = df.iloc[line,2:wide+2].values
                test_text_set = set(test_text)
                map_loc = {}
                for j in range(len(test_text)):
                    if test_text[j] not in map_loc:
                        map_loc[test_text[j]] = []
                    map_loc[test_text[j]].append(j)
                for i in range(fre_lines):
                    frequent_set = set(frequent_list[i][0])
                    frequent_inter_set = frequent_set & test_text_set
                    for inter_val in frequent_inter_set:
                        meta_val[i][map_loc[inter_val]] += 1

            meta_features_unique[csv_name_front] = meta_val.tolist()

            data_json = json.dumps(meta_features_unique[csv_name_front], cls=NpEncoder)
            fileObject = open(os.path.join(output_dir, csv_name_front + '.json'), 'w')
            fileObject.write(data_json)
            fileObject.close()
            del df
            gc.collect()
            
    data_json = json.dumps(meta_features_unique, cls=NpEncoder)
    fileObject = open(os.path.splitext(frequent_path)[0]+'-meta-features-fptree.json', 'w')
    fileObject.write(data_json)
    fileObject.close()

def normalization_half(meta_features_path, output_dir):
    metafeatures = np.load(metafeatures_path)
    metafeatures = (metafeatures + 1)/2
    np.save(os.path.join(output_dir, 'frequency_output_20-1-20-meta-features-half.npy'), metafeatures)

if __name__ == "__main__":

    minlength = 1
    minnum = 20
    
    allfiles_dir = "features_union_pre_with_noapi_features784"
    filtering_input_dir = os.path.join(allfiles_dir, 'frequency_output_20')
    filtering_output_dir = os.path.join(allfiles_dir, 'frequency_output_20-'+ str(minlength) + "-" + str(minnum))
    # freqres_filtering(allfiles_dir, filtering_input_dir, filtering_output_dir, minlength, minnum)

    front_frequent_path = filtering_output_dir + '.json'
    # unique_features(front_frequent_path)

    metafeatures_path = os.path.join(allfiles_dir, "frequency_output_20-1-20-meta-features.npy")
    sample_path = os.path.join(allfiles_dir, "vectors_noapi_features784.csv")
    
    # normalization_half(os.path.join(allfiles_dir, 'frequency_output_20-1-20-meta-features.npy'), allfiles_dir)