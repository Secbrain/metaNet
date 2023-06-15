# -*- coding: utf-8 -*-
import os
from lxml import etree
from threading import Thread
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
from scapy_ssl_tls.ssl_tls import TLS;
import random
# import dataframe_image as dfi
# from PIL import Image
import cv2
import pdb
import traceback

lie=[0,1,2,3,4,5,6,7,8,9,10,11,18,19,24,25,26,27,28,29,30,31,32,33,34,35]
tcp=[38,39,40,41,42,43,44,45,50,51]
udp=[40,41]

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

def extract_features(file_path):
    session_lens = []
    session_types = []
    session_times = []
    session_directions = []
    session_windows = []
    session_lensandtypes_val = {}

    # s_flags = {};
    #Fin Syn Reset Push Ack Urg EcE CWR
    session_flags = [];
    session_ip_flags = [];
    session_dport = [];
    session_fivetuples = [];

    flow = None
    try:
        flow = rdpcap(file_path)
    except:
        print(str(file_path) + " open error!")
        return None
    sessions = flow.sessions()
    for session in sessions:
        session_str = ""
        session_strs = session.split(' ')
        if len(session_strs) < 4:
            session_str = session
        else:
            # print(session_strs)
            if session_strs[1] < session_strs[3]:
                session_str = session
            else:
                session_str = session_strs[0] + " " + session_strs[3] + " " + session_strs[2] + " " + session_strs[1]
                if len(session_strs) > 4:
                    session_str = session_str + " " + " ".join(session_strs[4:])
        if session_str not in session_lensandtypes_val:
            session_lensandtypes_val[session_str] = []
        for packet in sessions[session]:
            _flags = [0, 0, 0, 0, 0, 0, 0, 0];
            type_val = None
            window = 0;
            if 'TCP' in packet:
                # print(str(bin(packet['TCP'].flags)))
                flags = list(reversed(list(str(bin(packet['TCP'].flags)))));
                flag_len = len(flags);
                for tt in range(flag_len):
                    if flags[tt] == '1':
                    	_flags[tt] += 1;
                # print(_flags)
            try:
                if 'TCP' in packet:
                    window = packet['TCP'].window;
                if 'SSL' in packet:
                    content_type = packet['SSL/TLS'].records[0].content_type
                    if content_type == 22:
                    	hand_type = packet['SSL/TLS'].records[0].type
                    	type_val = hand_type + content_type * 10
                    else:
                    	type_val = content_type
                elif packet.payload and packet.payload.name == 'IP':
                    type_val = packet.payload.proto
                else:
                    type_val = packet.type
            except:
                type_val = 1
            if type_val == None:
                type_val = 1000	
            #time,MTS,LBS,window,dst,dport
            dport = 0;
            if hasattr(packet,'dport')==True:
                dport = packet.dport;
            dst = '';
            _ip_flags = [0, 0, 0]
            if 'IP' in packet:
                dst = packet['IP'].dst;
                flags = list(reversed(list(str(bin(packet['IP'].flags)))));
                flag_len = len(flags);
                for tt in range(min(flag_len,2)):
                    if flags[tt] == '1':
                    	_ip_flags[tt] += 1;
                _ip_flags[2] += int(packet['IP'].frag);
                # print(_flags)
                
                # pdb.set_trace()
            else:
                dst = packet.dst;
            session_lensandtypes_val[session_str].append((packet.time,type_val,len(packet),window,dst,dport,_flags, _ip_flags)); 
            #session_flags.append(_flags);
            # s_flags[session_str] = ;
    
    for k,v in session_lensandtypes_val.items():
        if v == []:
            continue
        v1 = sorted(v)
        dport = v1[0][5];
        firstTime = v1[0][0];
        direction = v1[0][4];
        #direction
        temp = [];
        if v1[0][4]==direction:
            temp.append(0)
        else:
            temp.append(1)
        #flag
        flags_all = []
        flags_all.append(np.array(v1[0][6]))
        ip_flags_all = []
        ip_flags_all.append(np.array(v1[0][7]))
        lens = []
        lens.append(v1[0][2])
        mtss = []
        mtss.append(v1[0][1])
        timess = []
        timess.append(v1[0][0]-firstTime)
        windowss = []
        windowss.append(v1[0][3])
        for i in range(1,len(v1)):
            lens.append(v1[i][2])
            mtss.append(v1[i][1])
            timess.append(v1[i][0]-firstTime)
            windowss.append(v1[i][3])
            if v1[i][4]==direction:
                temp.append(0);
            else:
                temp.append(1);
            # flags_all = flags_all + np.array(v1[i][5])
            flags_all.append(np.array(v1[i][6]))
            ip_flags_all.append(np.array(v1[i][7]))

        session_dport.append(dport);
        session_directions.append(temp);
        session_flags.append(flags_all);
        session_ip_flags.append(ip_flags_all)
        session_lens.append(lens)
        session_types.append(mtss)
        session_times.append(timess);
        session_windows.append(windowss)
        session_fivetuples.append(k)
    return session_lens, session_types, session_windows, session_times, session_directions, session_dport, session_flags, session_ip_flags, session_fivetuples

def main():
    path_command = "DApp/pcap_original"
    output_json_path = "DApp/originaloutput"

    # features_all = []

    for class_name in os.listdir(path_command):
        class_path = os.path.join(path_command, class_name)
        for file_name in os.listdir(class_path):
            file_dir = os.path.join(class_path, file_name)

            file_json_save_path = os.path.join(file_dir,os.path.splitext(file_name)[0]+'_tuples.json')
            if os.path.exists(file_json_save_path):
                print(str(file_json_save_path) + "已存在！")
                continue

            file_path = os.path.join(file_dir, os.path.splitext(file_name)[0]+'_0.pcap')
            if os.path.exists(file_path):
                print(str(file_path) + "存在！")
                check_files = []
                for file_name_str in os.listdir(file_dir):
                    if os.path.splitext(file_name)[0] in file_name_str:
                        check_files.append(file_name_str)
                print(check_files)
                for file_name_str in check_files:
                    file_path = os.path.join(file_dir, file_name_str)

                    file_json_save_path = os.path.join(file_dir,os.path.splitext(file_name_str)[0]+'_tuples.json')
                    if os.path.exists(file_json_save_path):
                        print(str(file_json_save_path) + "已存在！")
                        continue

                    if not (os.path.isfile(file_path) and os.path.splitext(file_path)[1]=='.pcap'):
                        print(str(file_path) + "不是pcap文件！")
                        continue
                        
                    print(file_path + ' Processing!')
                    try:
                        lbs, session_types, window, time, direction, session_dport, session_flags, session_ip_flags, session_fivetuples = extract_features(file_path)

                        window_ts,forward_time_ts,backward_time_ts,forward_len_ts,backward_len_ts,duration_ts = [],[],[],[],[],[]
                        forward_window_ts, backward_window_ts = [],[]
                        forward_duration_ts, backward_duration_ts = [],[]
                        time_ts, len_ts = [],[]
                        flags_ts, forward_flags_ts, backward_flags_ts = [],[],[]
                        packets_num, forward_packets_num, backward_packets_num = [],[],[]
                        ip_flags_ts, ip_forward_flags_ts, ip_backward_flags_ts = [],[],[]
                        type_ts, forward_type_ts, backward_type_ts = [],[],[]
                        five_tuples = []
                        
                        all_len = len(lbs)
                        for pos in range(all_len):
                            window_mat = window[pos]
                            window_tuples = [np.max(window_mat),np.min(window_mat),np.mean(window_mat),np.std(window_mat)]
                            forward_time_lst,backward_time_lst = [],[]
                            forward_len_lst,backward_len_lst = [],[]
                            forward_window_lst,backward_window_lst = [],[]
                            forward_type_lst,backward_type_lst = [],[]
                            # print(type(direction[pos]))
                            
                            flags_all = np.array([0, 0, 0, 0, 0, 0, 0, 0])
                            forward_flags_all = np.array([0, 0, 0, 0, 0, 0, 0, 0])
                            backward_flags_all = np.array([0, 0, 0, 0, 0, 0, 0, 0])
                            
                            ip_flags_all = np.array([0, 0, 0])
                            ip_forward_flags_all = np.array([0, 0, 0])
                            ip_backward_flags_all = np.array([0, 0, 0])
                            
                            for nd,d in enumerate(direction[pos]):
                                if d==0:
                                    forward_time_lst.append(time[pos][nd])
                                    forward_len_lst.append(lbs[pos][nd])
                                    forward_window_lst.append(window_mat[nd])
                                    flags_all = flags_all + session_flags[pos][nd]
                                    forward_flags_all = forward_flags_all + session_flags[pos][nd]
                                    # print(session_ip_flags[pos][nd])
                                    ip_flags_all = ip_flags_all + session_ip_flags[pos][nd]
                                    ip_forward_flags_all = ip_forward_flags_all + session_ip_flags[pos][nd]
                                    forward_type_lst.append(session_types[pos][nd])
                                else:
                                    backward_time_lst.append(time[pos][nd])
                                    backward_len_lst.append(lbs[pos][nd])
                                    backward_window_lst.append(window_mat[nd])
                                    flags_all = flags_all + session_flags[pos][nd]
                                    backward_flags_all = backward_flags_all + session_flags[pos][nd]
                                    # print(session_ip_flags[pos][nd])
                                    ip_flags_all = ip_flags_all + session_ip_flags[pos][nd]
                                    ip_backward_flags_all = ip_backward_flags_all + session_ip_flags[pos][nd]
                                    backward_type_lst.append(session_types[pos][nd])
                            
                            packets_num.append(len(direction[pos]))
                            forward_packets_num.append(len(forward_len_lst))
                            backward_packets_num.append(len(backward_len_lst))
                            
                            type_ts.append(max(set(session_types[pos]), key=session_types[pos].count))
                            if len(forward_type_lst) > 1:
                                forward_type_ts.append(max(set(forward_type_lst), key=forward_type_lst.count))
                            else:
                                forward_type_ts.append(0)
                            if len(backward_type_lst) > 1:
                                backward_type_ts.append(max(set(backward_type_lst), key=backward_type_lst.count))
                            else:
                                backward_type_ts.append(0)

                            # tcp flag
                            flags_ts.append(flags_all.tolist()[:6])
                            forward_flags_ts.append(forward_flags_all.tolist()[:6])
                            backward_flags_ts.append(backward_flags_all.tolist()[:6])
                            
                            # ip flag
                            ip_flags_ts.append(ip_flags_all.tolist())
                            ip_forward_flags_ts.append(ip_forward_flags_all.tolist())
                            ip_backward_flags_ts.append(ip_backward_flags_all.tolist())
                            
                            if len(forward_time_lst) > 0:
                                forward_duration_ts.append(forward_time_lst[-1] - forward_time_lst[0])
                            else:
                                forward_duration_ts.append(0)
                            if len(backward_time_lst) > 0:
                                backward_duration_ts.append(backward_time_lst[-1] - backward_time_lst[0])
                            else:
                                backward_duration_ts.append(0)
                            
                            forward_time_mat = np.diff(forward_time_lst)
                            backward_time_mat = np.diff(backward_time_lst)
                            time_mat = np.diff(time[pos])
                            
                            if len(forward_window_lst) > 0:
                                forward_window_tuples = [np.max(forward_window_lst),np.min(forward_window_lst),np.mean(forward_window_lst),np.std(forward_window_lst)]
                            else:
                                forward_window_tuples = [0,0,0,0]
                            if len(backward_window_lst) > 0:
                                backward_window_tuples = [np.max(backward_window_lst),np.min(backward_window_lst),np.mean(backward_window_lst),np.std(backward_window_lst)]
                            else:
                                backward_window_tuples = [0,0,0,0]
                            
                            if len(time_mat)>0:
                                time_tuples = [np.max(time_mat),np.min(time_mat),np.mean(time_mat),np.std(time_mat)]
                            else:
                                time_tuples = [0,0,0,0]
                            if len(forward_time_mat)>0:
                                forward_time_tuples = [np.max(forward_time_mat),np.min(forward_time_mat),np.mean(forward_time_mat),np.std(forward_time_mat)]
                            else:
                                forward_time_tuples = [0,0,0,0]
                            if len(backward_time_mat)>0:
                                backward_time_tuples = [np.max(backward_time_mat),np.min(backward_time_mat),np.mean(backward_time_mat),np.std(backward_time_mat)]
                            else:
                                backward_time_tuples = [0,0,0,0]
                            if len(lbs[pos])>0:
                                len_tuples = [np.max(lbs[pos]),np.min(lbs[pos]),np.mean(lbs[pos]),np.std(lbs[pos])]
                            else:
                                len_tuples = [0,0,0,0]
                            if len(forward_len_lst)>0:
                                forward_len_tuples = [np.max(forward_len_lst),np.min(forward_len_lst),np.mean(forward_len_lst),np.std(forward_len_lst)]
                            else:
                                forward_len_tuples = [0,0,0,0]
                            if len(backward_len_lst)>0:
                                backward_len_tuples = [np.max(backward_len_lst),np.min(backward_len_lst),np.mean(backward_len_lst),np.std(backward_len_lst)]
                            else:
                                backward_len_tuples = [0,0,0,0]
                            
                            duration = max(time[pos])
                            duration_ts.append(duration)
                            
                            window_ts.append(window_tuples)
                            forward_window_ts.append(forward_window_tuples)
                            backward_window_ts.append(backward_window_tuples)
                            time_ts.append(time_tuples)
                            forward_time_ts.append(forward_time_tuples)
                            backward_time_ts.append(backward_time_tuples)
                            len_ts.append(len_tuples)
                            forward_len_ts.append(forward_len_tuples)
                            backward_len_ts.append(backward_len_tuples)
                            five_tuples.append(session_fivetuples[pos])
                            
                        data = {}
                        data['five_typles'] = five_tuples
                        data['length'] = lbs
                        data['type'] = session_types
                        data['window_tuples'] = window_ts
                        data['forward_window_tuples'] = forward_window_ts
                        data['backward_window_tuples'] = backward_window_ts
                        data['length_tuples'] = len_ts
                        data['forward_length_tuples'] = forward_len_ts
                        data['backward_length_tuples'] = backward_len_ts
                        data['time_tuples'] = time_ts
                        data['forward_time_tuples'] = forward_time_ts
                        data['backward_times_tuples'] = backward_time_ts
                        
                        data['packet_num'] = packets_num
                        data['forward_packet_num'] = forward_packets_num
                        data['backward_packet_num'] = backward_packets_num
                        data['duration'] = duration_ts
                        data['forward_duration'] = forward_duration_ts
                        data['backward_duration'] = backward_duration_ts
                        data['max_type'] = type_ts
                        data['forward_max_type'] = forward_type_ts
                        data['backward_max_type'] = backward_type_ts
                        
                        data['flags'] = flags_ts
                        data['forward_flags'] = forward_flags_ts
                        data['backward_flags'] = backward_flags_ts
                        
                        data['ip_flags'] = ip_flags_ts
                        data['forward_ip_flags'] = ip_forward_flags_ts
                        data['backward_ip_flags'] = ip_backward_flags_ts
                        
                        data['dport'] = session_dport

                        #input files
                        data_json = json.dumps(data)
                        fileObject = open(os.path.join(file_dir,os.path.splitext(file_name_str)[0]+'_tuples.json'),'w')
                        fileObject.write(data_json)
                        fileObject.close()
                        print(file_name+' output files!')

                    except Exception as e:
                        print(traceback.print_exc())
                        print(str(file_dir) + " file analysis error!")
            else:
                file_path = os.path.join(file_dir, 'ethereum_epoch0.pcap')
            
                if not (os.path.isfile(file_path) and os.path.splitext(file_path)[1]=='.pcap'):
                    print(str(file_path) + "is not a pcap file！")
                    continue

                print(file_path + ' Processing!')
                try:
                    lbs, session_types, window, time, direction, session_dport, session_flags, session_ip_flags, session_fivetuples = extract_features(file_path)

                    window_ts,forward_time_ts,backward_time_ts,forward_len_ts,backward_len_ts,duration_ts = [],[],[],[],[],[]
                    forward_window_ts, backward_window_ts = [],[]
                    forward_duration_ts, backward_duration_ts = [],[]
                    time_ts, len_ts = [],[]
                    flags_ts, forward_flags_ts, backward_flags_ts = [],[],[]
                    packets_num, forward_packets_num, backward_packets_num = [],[],[]
                    ip_flags_ts, ip_forward_flags_ts, ip_backward_flags_ts = [],[],[]
                    type_ts, forward_type_ts, backward_type_ts = [],[],[]
                    five_tuples = []
                    
                    all_len = len(lbs)
                    for pos in range(all_len):
                        window_mat = window[pos]
                        window_tuples = [np.max(window_mat),np.min(window_mat),np.mean(window_mat),np.std(window_mat)]
                        forward_time_lst,backward_time_lst = [],[]
                        forward_len_lst,backward_len_lst = [],[]
                        forward_window_lst,backward_window_lst = [],[]
                        forward_type_lst,backward_type_lst = [],[]
                        # print(type(direction[pos]))
                        
                        flags_all = np.array([0, 0, 0, 0, 0, 0, 0, 0])
                        forward_flags_all = np.array([0, 0, 0, 0, 0, 0, 0, 0])
                        backward_flags_all = np.array([0, 0, 0, 0, 0, 0, 0, 0])
                        
                        ip_flags_all = np.array([0, 0, 0])
                        ip_forward_flags_all = np.array([0, 0, 0])
                        ip_backward_flags_all = np.array([0, 0, 0])
                        
                        for nd,d in enumerate(direction[pos]):
                            if d==0:
                                forward_time_lst.append(time[pos][nd])
                                forward_len_lst.append(lbs[pos][nd])
                                forward_window_lst.append(window_mat[nd])
                                flags_all = flags_all + session_flags[pos][nd]
                                forward_flags_all = forward_flags_all + session_flags[pos][nd]
                                # print(session_ip_flags[pos][nd])
                                ip_flags_all = ip_flags_all + session_ip_flags[pos][nd]
                                ip_forward_flags_all = ip_forward_flags_all + session_ip_flags[pos][nd]
                                forward_type_lst.append(session_types[pos][nd])
                            else:
                                backward_time_lst.append(time[pos][nd])
                                backward_len_lst.append(lbs[pos][nd])
                                backward_window_lst.append(window_mat[nd])
                                flags_all = flags_all + session_flags[pos][nd]
                                backward_flags_all = backward_flags_all + session_flags[pos][nd]
                                # print(session_ip_flags[pos][nd])
                                ip_flags_all = ip_flags_all + session_ip_flags[pos][nd]
                                ip_backward_flags_all = ip_backward_flags_all + session_ip_flags[pos][nd]
                                backward_type_lst.append(session_types[pos][nd])
                        
                        packets_num.append(len(direction[pos]))
                        forward_packets_num.append(len(forward_len_lst))
                        backward_packets_num.append(len(backward_len_lst))
                        
                        type_ts.append(max(set(session_types[pos]), key=session_types[pos].count))
                        if len(forward_type_lst) > 1:
                            forward_type_ts.append(max(set(forward_type_lst), key=forward_type_lst.count))
                        else:
                            forward_type_ts.append(0)
                        if len(backward_type_lst) > 1:
                            backward_type_ts.append(max(set(backward_type_lst), key=backward_type_lst.count))
                        else:
                            backward_type_ts.append(0)

                        # tcp flag
                        flags_ts.append(flags_all.tolist()[:6])
                        forward_flags_ts.append(forward_flags_all.tolist()[:6])
                        backward_flags_ts.append(backward_flags_all.tolist()[:6])
                        
                        # ip flag
                        ip_flags_ts.append(ip_flags_all.tolist())
                        ip_forward_flags_ts.append(ip_forward_flags_all.tolist())
                        ip_backward_flags_ts.append(ip_backward_flags_all.tolist())
                        
                        if len(forward_time_lst) > 0:
                            forward_duration_ts.append(forward_time_lst[-1] - forward_time_lst[0])
                        else:
                            forward_duration_ts.append(0)
                        if len(backward_time_lst) > 0:
                            backward_duration_ts.append(backward_time_lst[-1] - backward_time_lst[0])
                        else:
                            backward_duration_ts.append(0)
                        
                        forward_time_mat = np.diff(forward_time_lst)
                        backward_time_mat = np.diff(backward_time_lst)
                        time_mat = np.diff(time[pos])
                        
                        if len(forward_window_lst) > 0:
                            forward_window_tuples = [np.max(forward_window_lst),np.min(forward_window_lst),np.mean(forward_window_lst),np.std(forward_window_lst)]
                        else:
                            forward_window_tuples = [0,0,0,0]
                        if len(backward_window_lst) > 0:
                            backward_window_tuples = [np.max(backward_window_lst),np.min(backward_window_lst),np.mean(backward_window_lst),np.std(backward_window_lst)]
                        else:
                            backward_window_tuples = [0,0,0,0]
                        
                        if len(time_mat)>0:
                            time_tuples = [np.max(time_mat),np.min(time_mat),np.mean(time_mat),np.std(time_mat)]
                        else:
                            time_tuples = [0,0,0,0]
                        if len(forward_time_mat)>0:
                            forward_time_tuples = [np.max(forward_time_mat),np.min(forward_time_mat),np.mean(forward_time_mat),np.std(forward_time_mat)]
                        else:
                            forward_time_tuples = [0,0,0,0]
                        if len(backward_time_mat)>0:
                            backward_time_tuples = [np.max(backward_time_mat),np.min(backward_time_mat),np.mean(backward_time_mat),np.std(backward_time_mat)]
                        else:
                            backward_time_tuples = [0,0,0,0]
                        if len(lbs[pos])>0:
                            len_tuples = [np.max(lbs[pos]),np.min(lbs[pos]),np.mean(lbs[pos]),np.std(lbs[pos])]
                        else:
                            len_tuples = [0,0,0,0]
                        if len(forward_len_lst)>0:
                            forward_len_tuples = [np.max(forward_len_lst),np.min(forward_len_lst),np.mean(forward_len_lst),np.std(forward_len_lst)]
                        else:
                            forward_len_tuples = [0,0,0,0]
                        if len(backward_len_lst)>0:
                            backward_len_tuples = [np.max(backward_len_lst),np.min(backward_len_lst),np.mean(backward_len_lst),np.std(backward_len_lst)]
                        else:
                            backward_len_tuples = [0,0,0,0]
                        
                        duration = max(time[pos])
                        duration_ts.append(duration)
                        
                        window_ts.append(window_tuples)
                        forward_window_ts.append(forward_window_tuples)
                        backward_window_ts.append(backward_window_tuples)
                        time_ts.append(time_tuples)
                        forward_time_ts.append(forward_time_tuples)
                        backward_time_ts.append(backward_time_tuples)
                        len_ts.append(len_tuples)
                        forward_len_ts.append(forward_len_tuples)
                        backward_len_ts.append(backward_len_tuples)
                        five_tuples.append(session_fivetuples[pos])
                        
                    data = {}
                    data['five_typles'] = five_tuples
                    data['length'] = lbs
                    data['type'] = session_types
                    data['window_tuples'] = window_ts
                    data['forward_window_tuples'] = forward_window_ts
                    data['backward_window_tuples'] = backward_window_ts
                    data['length_tuples'] = len_ts
                    data['forward_length_tuples'] = forward_len_ts
                    data['backward_length_tuples'] = backward_len_ts
                    data['time_tuples'] = time_ts
                    data['forward_time_tuples'] = forward_time_ts
                    data['backward_times_tuples'] = backward_time_ts
                    
                    data['packet_num'] = packets_num
                    data['forward_packet_num'] = forward_packets_num
                    data['backward_packet_num'] = backward_packets_num
                    data['duration'] = duration_ts
                    data['forward_duration'] = forward_duration_ts
                    data['backward_duration'] = backward_duration_ts
                    data['max_type'] = type_ts
                    data['forward_max_type'] = forward_type_ts
                    data['backward_max_type'] = backward_type_ts
                    
                    data['flags'] = flags_ts
                    data['forward_flags'] = forward_flags_ts
                    data['backward_flags'] = backward_flags_ts
                    
                    data['ip_flags'] = ip_flags_ts
                    data['forward_ip_flags'] = ip_forward_flags_ts
                    data['backward_ip_flags'] = ip_backward_flags_ts
                    
                    data['dport'] = session_dport

                    data_json = json.dumps(data)
                    fileObject = open(os.path.join(file_dir,os.path.splitext(file_name)[0]+'_tuples.json'),'w')
                    fileObject.write(data_json)
                    fileObject.close()
                    print(file_name+' ourput to dir!')

                except Exception as e:
                    print(traceback.print_exc())
                    print(str(file_dir) + " file analysis error!")

def pcap_json():
    path_command = "DApp/pcap_original"
    output_json_path = "DApp/originaloutput"

    # features_all = []

    for class_name in os.listdir(path_command):
        class_path = os.path.join(path_command, class_name)
        for file_name in os.listdir(class_path):
            file_dir = os.path.join(class_path, file_name)
            file_path = os.path.join(file_dir, 'ethereum_epoch0.pcap')
            if not (os.path.isfile(file_path) and os.path.splitext(file_path)[1]=='.pcap'):
                print(str(file_path) + " is not a pcap file！")
                continue
                
            file_json_save_path = os.path.join(file_dir,os.path.splitext(file_name)[0]+'_tuples.json')
            if os.path.exists(file_json_save_path):
                print(str(file_json_save_path) + " has been existed!")
                continue

            file_json_save_path = os.path.join(file_dir,os.path.splitext(file_name)[0]+'_pcap.json')
            if os.path.exists(file_json_save_path):
                print(str(file_json_save_path) + " has been existed!")
                continue

            print(file_path + ' Processing!')
            try:
                lbs, session_types, window, time, direction, session_dport, session_flags, session_ip_flags, session_fivetuples = extract_features(file_path)
                    
                data = {}
                data['lbs'] = lbs
                data['session_types'] = session_types
                data['window'] = window
                data['time'] = time
                data['direction'] = direction
                data['session_dport'] = session_dport
                data['session_flags'] = session_flags
                data['session_ip_flags'] = session_ip_flags
                data['session_fivetuples'] = session_fivetuples

                data_json = json.dumps(data)
                fileObject = open(os.path.join(file_dir,os.path.splitext(file_name)[0]+'_pcap.json'),'w')
                fileObject.write(data_json)
                fileObject.close()

            except Exception as e:
                print(traceback.print_exc())
                print(str(file_dir) + "error!")

def json_to_csv():
    path_command = "DApp/pcap_original"
    output_json_path = "DApp/originaloutput"

    features_all = []

    for class_name in os.listdir(path_command):
        class_path = os.path.join(path_command, class_name)
        for file_name in os.listdir(class_path):
            file_dir = os.path.join(class_path, file_name)
            
            for file_name_str in os.listdir(file_dir):
                if "_tuples.json" not in file_name_str:
                    continue

                file_json_save_path = os.path.join(file_dir,file_name_str)

                print(file_json_save_path + ' Processing!')
                try:
                    data = None
                    with open(file_json_save_path) as fp:
                        data = json.load(fp)
                    
                    five_tuples = data['five_typles']
                    lbs = data['length'] 
                    session_types = data['type']
                    window_ts = data['window_tuples']
                    forward_window_ts = data['forward_window_tuples']
                    backward_window_ts = data['backward_window_tuples']
                    len_ts = data['length_tuples']
                    forward_len_ts = data['forward_length_tuples']
                    backward_len_ts = data['backward_length_tuples']
                    time_ts = data['time_tuples']
                    forward_time_ts = data['forward_time_tuples']
                    backward_time_ts = data['backward_times_tuples']
                    
                    packets_num = data['packet_num']
                    forward_packets_num = data['forward_packet_num']
                    backward_packets_num = data['backward_packet_num']
                    duration_ts = data['duration']
                    forward_duration_ts = data['forward_duration']
                    backward_duration_ts = data['backward_duration']
                    type_ts = data['max_type']
                    forward_type_ts = data['forward_max_type']
                    backward_type_ts = data['backward_max_type']
                    
                    flags_ts = data['flags']
                    forward_flags_ts = data['forward_flags']
                    backward_flags_ts = data['backward_flags']
                    
                    ip_flags_ts = data['ip_flags']
                    ip_forward_flags_ts = data['forward_ip_flags']
                    ip_backward_flags_ts = data['backward_ip_flags']
                    
                    session_dport = data['dport']

                    # flags_all = session_flags
                    dport_all = session_dport
                    if len(window_ts) < 5:
                        continue

                    for i in range(len(window_ts)):
                        features_val = []
                        features_val.append(class_name) 
                        features_val.append(file_name) 
                        features_val.append(five_tuples[i])
                        features_val.extend(window_ts[i])
                        features_val.extend(forward_window_ts[i])
                        features_val.extend(backward_window_ts[i])
                        features_val.extend(len_ts[i])
                        features_val.extend(forward_len_ts[i])
                        features_val.extend(backward_len_ts[i])
                        features_val.extend(time_ts[i])
                        features_val.extend(forward_time_ts[i])
                        features_val.extend(backward_time_ts[i])
                        
                        features_val.append(packets_num[i])
                        features_val.append(forward_packets_num[i])
                        features_val.append(backward_packets_num[i])
                        features_val.append(duration_ts[i])
                        features_val.append(forward_duration_ts[i])
                        features_val.append(backward_duration_ts[i])
                        features_val.append(type_ts[i])
                        features_val.append(forward_type_ts[i])
                        features_val.append(backward_type_ts[i])
                        
                        features_val.extend(flags_ts[i])
                        features_val.extend(forward_flags_ts[i])
                        features_val.extend(backward_flags_ts[i])
                        
                        features_val.extend(ip_flags_ts[i])
                        features_val.extend(ip_forward_flags_ts[i])
                        features_val.extend(ip_backward_flags_ts[i])
                        
                        # features_val.append(dport_all[i])
                        
                        features_all.append(features_val)
                except Exception as e:
                    print(traceback.print_exc())
                    print(str(file_dir) + " file analysis error!")

    features_vector = ['category', 'name', 'five_tuples', 
    	'window_max', 'window_min', 'window_mean', 'window_std', 
    	'forward_window_max', 'forward_window_min', 'forward_window_mean', 'forward_window_std', 
    	'backward_window_max', 'backward_window_min', 'backward_window_mean', 'backward_window_std', 
    	'len_ts_max', 'len_ts_min', 'len_ts_mean', 'len_ts_std',
    	'forward_len_ts_max', 'forward_len_ts_min', 'forward_len_ts_mean', 'forward_len_ts_std',
    	'backward_len_ts_max', 'backward_len_ts_min', 'backward_len_ts_mean', 'backward_len_ts_std',
        'time_ts_max', 'time_ts_min', 'time_ts_mean', 'time_ts_std',
        'forward_time_ts_max', 'forward_time_ts_min', 'forward_time_ts_mean', 'forward_time_ts_std',
        'backward_time_ts_max', 'backward_time_ts_min', 'backward_time_ts_mean', 'backward_time_ts_std',
        'packet_num', 'forward_packet_num', 'backward_packet_num',
        'duration', 'forward_duration', 'backward_duration',
        'max_type', 'forward_max_type', 'backward_max_type',
        'flags_fin', 'flags_syn', 'flags_res', 'flags_pus', 'flags_ack', 'flags_urg', 
        'forward_flags_fin', 'forward_flags_syn', 'forward_flags_res', 'forward_flags_pus', 'forward_flags_ack', 'forward_flags_urg', 
        'backward_flags_fin', 'backward_flags_syn', 'backward_flags_res', 'backward_flags_pus', 'backward_flags_ack', 'backward_flags_urg', 
        'ip_flags_mf', 'ip_flags_df', 'ip_offset', 
        'forward_ip_flags_mf', 'forward_ip_flags_df', 'forward_ip_offset', 
        'backward_ip_flags_mf', 'backward_ip_flags_df', 'backward_ip_offset']
    vector_val_df = pd.DataFrame(features_all, columns = features_vector)
    vector_val_df.to_csv(os.path.join(output_json_path, 'featuresall.csv'),quoting=1,index=False)
    print('featuresall.csv output successfully!')

def split_pcap_files():
    path_command = r"DApp/pcap_original"
    output_json_path = r"DApp/pcap_original/originaloutput"

    # features_all = []

    for class_name in os.listdir(path_command):
        class_path = os.path.join(path_command, class_name)
        for file_name in os.listdir(class_path):
            file_dir = os.path.join(class_path, file_name)
            file_path = os.path.join(file_dir, 'ethereum_epoch0.pcap')
            if not (os.path.isfile(file_path) and os.path.splitext(file_path)[1]=='.pcap'):
                print(str(file_path) + " is not a pcap file!")
                continue
             
            file_json_save_path = os.path.join(file_dir,os.path.splitext(file_name)[0]+'_tuples.json')
            if os.path.exists(file_json_save_path):
                print(str(file_json_save_path) + " has been existed!")
                continue

            file_json_save_path = os.path.join(file_dir,os.path.splitext(file_name)[0]+'_0.pcap')
            if os.path.exists(file_json_save_path):
                print(str(file_json_save_path) + " has been existed!")
                continue

            filesize = os.path.getsize(file_path)/1024/1024
            if filesize < 500:
                print(file_path + ' lower than 500MB!')
                continue

            print(file_path + ' Processing!')
            flow = None
            try:
                flow = rdpcap(file_path)
            except:
                print(str(file_path) + " analysis error!")
                return None
            session_packets = {}
            sessions = flow.sessions()
            for session in sessions:
                session_str = ""
                session_strs = session.split(' ')
                if len(session_strs) < 4:
                    session_str = session
                else:
                    # print(session_strs)
                    if session_strs[1] < session_strs[3]:
                        session_str = session
                    else:
                        session_str = session_strs[0] + " " + session_strs[3] + " " + session_strs[2] + " " + session_strs[1]
                        if len(session_strs) > 4:
                            session_str = session_str + " " + " ".join(session_strs[4:])
                
                if session_str not in session_packets:
                    session_packets[session_str] = []
                session_packets[session_str].extend(sessions[session])

            part_num = int(filesize / 500) + 1
            
            lens_all = 0
            for k,v in session_packets.items():
                if v == []:
                    continue
                lens_all += len(v)
            
            part_length = lens_all/part_num
            
            packets_file = []
            mm_sum = 0

            iii_index = 0
            for k,v in session_packets.items():
                if v == []:
                    continue
                packets_file.extend(v)
                mm_sum += len(v)

                if mm_sum >= part_length:
                    output_json_part_path = os.path.join(file_dir, os.path.splitext(file_name)[0] + "_" + str(iii_index) + ".pcap")
                    wrpcap(output_json_part_path,packets_file)
                    packets_file = []
                    mm_sum = 0
                    iii_index += 1
                    print('split to save')

            if len(packets_file) > 0:
                output_json_part_path = os.path.join(file_dir, os.path.splitext(file_name)[0] + "_" + str(iii_index) + ".pcap")
                wrpcap(output_json_part_path,packets_file)
                packets_file = []
                mm_sum = 0

def mergejson_files_lbsandmts():
    path_command = sys.argv[1]
    output_json_dir = sys.argv[2]
    output_json_name = sys.argv[3]

    min_length = 2
    max_packet_length = 6000

    filenames = [os.path.splitext(filename)[0] for filename in os.listdir(path_command) \
                 if os.path.isfile(os.path.join(path_command, filename)) and os.path.splitext(filename)[1]=='.json']
    filenames = sorted(filenames)
    filenames_map = {}
    for i in range(len(filenames)):
        filenames_map[filenames[i]] = i

    data_json = json.dumps(filenames_map)
    fileObject = open(os.path.join(output_json_dir,'filenames_to_ids.json'),'w')
    fileObject.write(data_json)
    fileObject.close()

    datasets = []

    for file_name in os.listdir(path_command):
        file_path = os.path.join(path_command, file_name)
        if not (os.path.isfile(file_path) and os.path.splitext(file_path)[1]=='.json'):
            print(str(file_path) + " is not a json file!")
            continue

        fileid = filenames_map[os.path.splitext(file_name)[0]]

        json_val = None
        with open(file_path) as fp:
            json_val = json.load(fp)

        for i in range(len(json_val['length'])):
            # 过滤一波
            if len(json_val['length'][i]) < min_length:
                continue
            flow = [ix if ix <= max_packet_length else max_packet_length for ix in json_val['length'][i]]
            datasets.append({'label':fileid, 'length': flow, 'type': json_val['type'][i]})

    data_json = json.dumps(datasets)
    fileObject = open(os.path.join(output_json_dir, output_json_name),'w')
    fileObject.write(data_json)
    fileObject.close()
    print(output_json_name+' output successfully!')

def mergejson_files_lbsandmt_twoclass():
    path_command = sys.argv[1]
    output_json_dir = sys.argv[2]
    output_json_name = sys.argv[3]

    min_length = 2
    max_packet_length = 6000

    datasets = []

    for file_name in os.listdir(path_command):
        file_path = os.path.join(path_command, file_name)
        if not (os.path.isfile(file_path) and os.path.splitext(file_path)[1]=='.json'):
            print(str(file_path) + " is not a json file!")
            continue

        if os.path.splitext(file_name)[0] != 'Benign_tuples':
            fileid = 1
        else:
            fileid = 0

        json_val = None
        with open(file_path) as fp:
            json_val = json.load(fp)

        for i in range(len(json_val['length'])):
            if len(json_val['length'][i]) < min_length:
                continue
            flow = [ix if ix <= max_packet_length else max_packet_length for ix in json_val['length'][i]]
            datasets.append({'label':fileid, 'length': flow, 'type': json_val['type'][i]})

    data_json = json.dumps(datasets)
    fileObject = open(os.path.join(output_json_dir, output_json_name),'w')
    fileObject.write(data_json)
    fileObject.close()
    print(output_json_name+' output successfully!')

if __name__ == '__main__':
    split_pcap_files()
    main()
    pcap_json()
    json_to_csv()
    mergejson_files_lbsandmts()
    # mergejson_files_lbsandmt_twoclass()
