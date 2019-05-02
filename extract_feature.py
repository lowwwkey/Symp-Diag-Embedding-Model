#!/usr/bin/env python3
#-*-coding:utf-8-*-

'''
@File       : extract_feature.py
@Discription: 提取11个科室的特征，得到特征向量
@Author     : Guangkai Li
@Date:      : 2017/04/29
'''

import heapq
import numpy as np
import pandas as pd
import random
import re
from collections import defaultdict, Counter
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split

 
def load_data(file_name):
    """
    导入数据
    Args:
        file_name: excel文件名字
    """
    df = pd.read_excel(file_name, header=1)
    return df[df.病历内容.notnull()]

def emr_parts(df, min_num=35): # 儿科40：，黏膜：35，颌面：40，激光：35，急诊：40，修复：20，特修复：40，牙体：40，正畸：40，牙周：20，种植：25
    """
    将病历内容以列表存储
    Args:
        df:dateframe数据
    Returns:
        emr_lst:['病历内容1','病历内容2',...]
    """
    emr_co = list(df.病历内容)
    emr_lst = []

    for emr in emr_co: 
    	cc_re = re.compile(r'(主  诉：)(.*)')
    	cc_search = cc_re.search(emr)
    
    	hpi_re = re.compile(r'(现病史：)(.*)')
    	hpi_search = hpi_re.search(emr)    
    
    	ph_re = re.compile(r'(既往史：)(.*)')
    	ph_search = ph_re.search(emr)
    
    	fh_re = re.compile(r'(家族史：)(.*)')
    	fh_search = fh_re.search(emr)
    
    	pe_re = re.compile(r'(检  查：)((.|\n)*)\n表格')
    	pe_search = pe_re.search(emr)
    
    	diag_re = re.compile(r'(诊断：)(\s*)(.*)')
    	diag_search = diag_re.search(emr)
    
    	if pe_search != None and diag_search != None:
            diag_match = diag_search.group(3)
            s = ['1', '?', '？']
            # s = []
            if all(t not in diag_match for t in s):
            	cc_match = cc_search.group(2)
            	hpi_match = hpi_search.group(2)
            	ph_match = ph_search.group(2)
            	fh_match = fh_search.group(2)
            	pe_match = pe_search.group(2)
            	emr_lst.append([cc_match, hpi_match, ph_match, fh_match, pe_match, diag_match])
    diag = [emr[-1] for emr in emr_lst]
    diag_count = Counter(diag)

    return [emr for emr in emr_lst if diag_count[emr[-1]] >= min_num and '唇炎' not in emr[-1]]

def to_extract(emr_lst, min_num=30):
    """
    返回待提取的病历各部分内容
    Args:
        emr_lst
        min_num: 最少病历数量
    Returns:
        病历各部分内容，{'诊断1':{'cc':[...], 'hpi':[...], ...}, ...}
    """
    cl_emr = {}
    diag = []
    for emr in emr_lst:
        cl_emr[emr[5]] = cl_emr.get(emr[5], defaultdict(list))
        cl_emr[emr[5]]['cc'].append(emr[0])
        cl_emr[emr[5]]['hpi'].append(emr[1])
        cl_emr[emr[5]]['ph'].append(emr[2])
        cl_emr[emr[5]]['fh'].append(emr[3])
        cl_emr[emr[5]]['pe'].append(emr[4])
        diag.append(emr[5])
                                        
    # diag_count = Counter(diag)
    # diag_count = sorted(diag_count.items(), key=lambda item: item[1], reverse=True)

    # return {key: value for key, value in cl_emr.items() if key in [i[0] for i in diag_count if i[1]>min_num]}
    return cl_emr

def extract_feature(to_extract_emr, min_freq=5):
    """
    提取特征
    Args:
        to_extract_emr
        min_freq:最小频次
    Returns:
        特征, [feature1, feature2, ...]
    """
    cc_lst, hpi_lst, ph_lst, fh_lst, pe_lst = [], [], [], [], []
    cc_symp, hpi_symp, ph_symp, fh_symp, pe_symp = {}, {}, {}, {}, {}
    for diag in to_extract_emr:
        cc_lst += to_extract_emr[diag]['cc']
        hpi_lst += to_extract_emr[diag]['hpi']
        ph_lst += to_extract_emr[diag]['ph']
        fh_lst += to_extract_emr[diag]['fh']
        pe_lst += to_extract_emr[diag]['pe']
    # print(cc_lst)
    # ---------------主诉部分------------------
    # ---黏膜科---
    patterns = [r'(口腔溃疡反复发作)(.*)', r'.*(颊黏膜色白网纹)(.*)', r'.*(正畸转诊洁治)(.*)', r'.*(要求洁治)(.*)'", "r'.*(转诊主诉)(.*)', r'.*(正畸科转诊牙周治疗)(.*)', r'.*(唇(反复)?脱皮)(.*)']
    # ---儿科---
    # patterns = [r'(要求拔除)(.*)', r'(要求治疗)(.*)', r'(要求镇静下治疗)(.*)', r'发现.*牙(.*)([半|一|\d].*)', r'.*牙(.*)([半|一|\d].*)']
    # ---颌面外科---
    # patterns = [r'(要求拔除)(.*)?', r'.*(转诊拔牙).*', r'.*(牙龈肿痛).*', r'(转诊拔牙).*', r'.*牙(.*)([半|一|\d].*)']
    # ---激光科---
    # patterns = [r'(要求拔除)(.*)', r'.*齿(.*)([半|一|\d].*)', r'.*.?.?.?(发现膨隆)(.*)', r'.*?(发现起包).*', r'.*(正畸转诊拔牙).*', r'.*?(发现肿物).*']
    # ---急诊科---
    # patterns = [r'.*(牙龈肿胀疼痛)([半|一|\d].*)', '.*(牙(龈)?(肿|疼)痛)([半|一|\d].*)', '.*(牙热痛冷缓解)([半|一|\d].*)', '.*(牙自发痛).*']
    # ---修复科---
    # patterns = [r'.*(牙(缺损|折断|劈裂|冷刺激痛))([半|一|\d]).*', '.*(要求修复).*']
    # ---特修复---
    # patterns = [r'.*(牙(缺损|折断|劈裂|冷刺激痛|缺失|松动))([半|一|\d]).*', '.*(要求修复).*']
    # ---牙体---
    # patterns = [r'.*(牙(冷热通|有洞|咬合疼痛|自发痛|疼痛|夜间痛|咬合不适|冷水刺激疼痛|冷刺激敏感|食物嵌塞|发现有龋|折断|填体脱落|牙龈肿胀))([数|半|一|\d]).*']
    # ---正畸---
    # patterns = [r'.*(牙未长出).*', r'.*(牙不齐).*', r'.*(嘴突).*', r'.*(地包天).*']
    # ---牙周---
    # patterns = [r'.*(无症状主诉).*', r'.*(有症状主诉).*', r'.*(要求洗牙).*']
    # ---种植---
    # patterns = [r'.*牙(.*)([半|一|\d].*)']
    for cc in cc_lst:
        for p in patterns:
            re_ = re.compile(p)
            search_ = re_.search(cc.strip())
            if search_ != None:
                match_ = search_.group(1)
                cc_symp[match_] = cc_symp.get(match_, 0) + 1
    
    # --------------现病史部分-----------------
    max_n = max(len(hpi) for hpi in hpi_lst)
    d = {}
    for j in range(2, max_n):
        hpi_extract = Counter()
        for hpi in hpi_lst:
            hpi = hpi.strip()
            s = ['\t',',','.',':',';','，','。',':','；','!','！','、','“', '"']
            hpi_extract.update(Counter([hpi[i:i+j] for i in range(len(hpi)-j+1) if all(t not in hpi[i:i+j] for t in s)]))
        if len(hpi_extract) == 0:
            break
        d[j] = hpi_extract
        if j>2:
           tmp = []
           for fea in d[j-1]:
               n = 0
               for fea_ in hpi_extract:
                   if fea in fea_:
                       n += hpi_extract[fea_]
               if d[j-1][fea] <= n+5:
                   tmp.append(fea)
           for fea in tmp:
               d[j-1].pop(fea)
    for i in d:
        hpi_symp.update({key: value for key, value in d[i].items()})

    #--------------既往史、家族史和检查部分--------------
    def split_symp(lst):
        tmp = Counter()
        split_delimiter = r'\n|，|；|。|\t|,|;|；|!|！|、'
        for i in lst:
            lst = re.split(split_delimiter, i.strip())
            lst = [symp.strip() for symp in lst if len(symp) > 1]
            tmp.update(lst)
        return tmp

    ph_symp = split_symp(ph_lst)
    fh_symp = split_symp(fh_lst)
    pe_symp = split_symp(pe_lst)

    cc_symp = [key for key, value in cc_symp.items() if value >= min_freq]
    hpi_symp = [key for key, value in hpi_symp.items() if value >= min_freq]
    ph_symp = [key for key, value in ph_symp.items() if value >= min_freq]
    fh_symp = [key for key, value in fh_symp.items() if value >= min_freq]
    pe_symp = [key for key, value in pe_symp.items() if value >= min_freq]

    symp_lst = cc_symp + hpi_symp + ph_symp + fh_symp + pe_symp
    # with open("./data/symp_lst_new.txt", "w") as f:
    #     for i in symp_lst:
    #         f.write(i+"\n")
    
    return [cc_symp, hpi_symp, ph_symp, fh_symp, pe_symp]

def feature_vector(emr_lst, symp):
    """
    one-hot表示
    """
    def to_vec(emr_part, symp_part):
        v = []
        for i in symp_part:
            if i in emr_part:
                v.append(1)
            else:
                v.append(0)
        return v

    symp_vec = []
    for emr in emr_lst:
        vec = []
        for i in range(len(emr[:-1])):
            vec += to_vec(emr[i], symp[i])
        symp_vec.append(vec)

    diag = [emr[-1] for emr in emr_lst]
    diag_index = {}
    for i in diag:
        if i not in diag_index:
            diag_index[i] = len(diag_index)
    diag_vec = [diag_index[i] for i in diag]

    return np.array(symp_vec), np.array(diag_vec)
    
def split_vec(symp_vec, diag_vec):
    random.seed(3)
    ind = random.sample(range(len(symp_vec)), int(0.8*len(symp_vec)))
    training_vec, training_label, test_vec, test_label = [], [], [], []
    for i in range(len(symp_vec)):
        if i in ind:
            training_vec.append(symp_vec[i])
            training_label.append(diag_vec[i])
        else:
            test_vec.append(symp_vec[i])
            test_label.append(diag_vec[i])
    return training_vec, training_label, test_vec, test_label

def split_dataset(symp_vec, diag_vec):
    training_vec, test_vec, training_label, test_label = train_test_split(symp_vec, diag_vec, test_size=0.2, stratify=diag_vec, random_state=7)
    return training_vec, training_label, test_vec, test_label

if __name__ == '__main__':
    df = load_data('./data/data.xlsx')
    emr_lst = emr_parts(df)
    to_extract_emr = to_extract(emr_lst)
    symp = extract_feature(to_extract_emr)

    symp_vec, diag_vec = feature_vector(emr_lst, symp)
    training_vec, training_label, test_vec, test_label = split_dataset(symp_vec, diag_vec)

    # mi = mutual_info_classif(training_vec, training_label, discrete_features='auto', n_neighbors=7, copy=True, random_state=3)
    # mi = mi.tolist()
    # max_mi_index = list(map(mi.index, heapq.nlargest(100, mi)))
    # max_mi_index = sorted(max_mi_index)
    # print(max_mi_index)
    # training_vec = np.array([v[max_mi_index] for v in training_vec])
    # test_vec = np.array([v[max_mi_index] for v in test_vec])

    X_train,y_train,X_test,y_test = [],[],[],[]
    for i in range(len(training_label)):
        if sum(training_vec[i]) > 0:
            X_train.append(training_vec[i])
            y_train.append(training_label[i])
    for i in range(len(test_label)):
        if sum(test_vec[i]) > 0:
            X_test.append(test_vec[i])
            y_test.append(test_label[i])

    training_vec, training_label, test_vec, test_label = X_train,y_train,X_test,y_test

    np.save('./data/training_vec.npy', training_vec)
    np.save('./data/training_label.npy', training_label)
    np.save('./data/test_vec.npy', test_vec)
    np.save('./data/test_label.npy', test_label)
    print(test_label)