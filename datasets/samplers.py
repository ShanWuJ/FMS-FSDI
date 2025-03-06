import os
import torch
import math
import numpy as np
from copy import deepcopy
from torch.utils.data import Sampler

# sampler used for meta-training
class meta_batchsampler(Sampler):
    
    def __init__(self,data_source,way,shots):

        self.way = way
        self.shots = shots

        class2id = {}

        for i,(image_path,class_id) in enumerate(data_source.imgs):
            if class_id not in class2id:
                class2id[class_id]=[]
            class2id[class_id].append(i)

        self.class2id = class2id


    def __iter__(self):

        temp_class2id = deepcopy(self.class2id)
        for class_id in temp_class2id:
            np.random.shuffle(temp_class2id[class_id])       

        while len(temp_class2id) >= self.way:  #一直循环到少于way个类别为止

            id_list = []

            list_class_id = list(temp_class2id.keys())

            pcount = np.array([len(temp_class2id[class_id]) for class_id in list_class_id])

            batch_class_id = np.random.choice(list_class_id,size=self.way,replace=False,p=pcount/sum(pcount))  #随机选10类

            for shot in self.shots:  # [5,15]
                for class_id in batch_class_id:
                    for _ in range(shot):
                        id_list.append(temp_class2id[class_id].pop())    #先针对shot选，共有10-way*5shot张 再对query-shot选，选10-way*15-query shot张，因此这个列表一次有10*5+10*15张

            for class_id in batch_class_id:
                if len(temp_class2id[class_id])<sum(self.shots):
                    temp_class2id.pop(class_id)

            yield id_list


# sampler used for meta-testing
class random_sampler(Sampler):

    def __init__(self,data_source,way,shot,query_shot=16,trial=1000):   #这里的1000

        class2id = {}

        for i,(image_path,class_id) in enumerate(data_source.imgs):
            if class_id not in class2id:
                class2id[class_id]=[]
            class2id[class_id].append(i)    #对应的类别的图像索引

        self.class2id = class2id
        self.way = way         # 5
        self.shot = shot       # 5
        self.trial = trial     # 1000
        self.query_shot = 16

    def __iter__(self):

        way = self.way
        shot = self.shot
        trial = self.trial
        query_shot = self.query_shot
        
        class2id = deepcopy(self.class2id)        
        list_class_id = list(class2id.keys())    #类别

        for i in range(trial):  #1000次循环

            id_list = []
 
            np.random.shuffle(list_class_id)
            picked_class = list_class_id[:way]   #取5类

            for cat in picked_class:
                np.random.shuffle(class2id[cat])  #打乱取出的5类的图像索引
                
            for cat in picked_class:
                id_list.extend(class2id[cat][:shot])   #每循环一次取5张，一共有5类 即这一个之后，就得到了按类别放在一起的5*5张图像的索引  [00,01,02,03,04,10,11,12,13,14,20,21,22,23,24,30,31,32,33,34,41,42,43,44] 支持集
            for cat in picked_class:
                id_list.extend(class2id[cat][shot:(shot+query_shot)])  #同样得到5-way*15-query_shot张图像的索引

            yield id_list   #重复1000次，最后列表长度是：(5-way*5shot+5-way*15query-shot)*1000