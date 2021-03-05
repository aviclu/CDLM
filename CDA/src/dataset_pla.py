from torch.utils.data.dataset import Dataset
import torch
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import sys
import os
from transformers import *
import random
import string
#For loading large text corpora

maxInt = sys.maxsize
maxInt = int(maxInt/10)
decrement = True
while decrement:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.
    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True

printable = set(string.printable)
remove_list = ['\n', '\x0b', '\x0c', '\r', '\t']
for i in remove_list:
    printable.remove(i)
    #print(printable)
printable.add('\001')

def clean(abstract):
    #line = text.replace('\0','')
    abstract = abstract.strip()
    abstract = abstract.replace('/n', '')
    abstract = ''.join(filter(lambda x: x in printable, abstract))
    return abstract

def process(text):
    text_array = []
    for section in text.split('\t'):
        section = clean(section)
        if section != '':
            text_array.append(section)
    return text_array

class MyDataset(Dataset):
    def __init__(self, data_path, pos_path, max_len=18):
        super(MyDataset, self).__init__()
        preprocess_text = data_path[:-4]+'.npy'
        self.sentences = np.load(preprocess_text, mmap_mode='r')
        self.max_len = max_len
        #sentences = sentences.astype(np.float16)
        #np.save(f'{data_path[:-4]}.snpy', sentences)
        bd_file = data_path[:-4]+'.index'
        bd_list = []
        texts = []
        labels = []
        pos = []
        pos_dict = {}
        bd_pairs = []
        
        with open(pos_path) as f:
            for i in f:
                i = i.strip()
                i = i.split(',')
                pos_dict[int(i[0])] =[int(j) for j in i[1:]] 
 
        with open(bd_file) as f:
            count = 0
            for i in f:
                count += int(i.strip())
                bd_list.append(count)

        with open(data_path) as csv_file:
            reader = csv.reader(csv_file, quotechar='"')
            bd_next = 0
            for idx, line in enumerate(reader):
                cite_pos = []
                bd = bd_list[2*idx]
                label = int(line[0])
                text_1 = (bd_next,bd)
                bd_next = bd_list[2*idx+1]
                text_2 = (bd,bd_next)
                bd_pairs.append((text_1, text_2))
                labels.append(label)
                pos.append(pos_dict.get(idx,[]))

        print(len(labels))
        labels = np.array(labels)
        self.texts = bd_pairs
        self.labels = labels
        self.pos = pos
        self.mask = []

    def mismatch(self, labels, bd_pairs):
        m_labels = []
        m_bd_pairs = []
        num_exs = len(labels) 
        labels_x = labels[:int(0.2*num_exs)]
        bd_pairs_x = bd_pairs[:int(0.2*num_exs)]  
        for i,j in zip(labels[int(0.2*num_exs):], bd_pairs[int(0.2*num_exs):]):
            m_labels.append(i)
            m_bd_pairs.append(j)
            m_labels.append(0)
            text_1 = j[0]
            m_text_2 = random.choice(bd_pairs_x)[1]
            m_bd_pairs.append((text_1, m_text_2))
        return m_labels, m_bd_pairs
        
    def get_mask(self):
        return np.array(self.mask)
            
    def get_pos(self):
        return self.pos

    def __len__(self):
        return len(self.labels)

    def process(self, bd, cite_pos):
        i = self.sentences[bd[0]:bd[1]]
        i = np.array(i)
        max_length = self.max_len
        if i.shape[0] < max_length:
            padding = np.zeros((max_length-i.shape[0],1024))
            doc = np.concatenate((i, padding))
            mask = [1]*i.shape[0]+[0]*(max_length-i.shape[0])
            new_pos = cite_pos
        else:
            win_front = cite_pos-random.randint(0,int(max_length/2))
            win_front = max(0,win_front)
            win_back = cite_pos+random.randint(2,int(max_length/2))
            doc = i[cite_pos:cite_pos+1]
            doc = np.concatenate((i[win_front:cite_pos], doc, i[cite_pos+1:win_back]))
            new_pos = cite_pos-win_front
            mask = [1]*max_length
            if doc.shape[0] < max_length:
                padding = np.zeros((max_length-doc.shape[0],1024))
                mask = [1]*doc.shape[0]+[0]*(max_length-doc.shape[0])
                doc = np.concatenate((doc, padding))
        self.mask.append(mask)
        return doc, new_pos
        
    def __getitem__(self, index):
        cite_pos = self.pos[index]
        label = self.labels[index]
        text_1, np_1 = self.process(self.texts[index][0], cite_pos[0])
        text_2, np_2 = self.process(self.texts[index][1], cite_pos[1])
        #cls_ = self.process(text_1, text_2)
        self.pos[index] = (np_1, np_2)
        cls1 = text_1
        cls2 = text_2
        return cls1.astype(np.float32), cls2.astype(np.float32), label, cite_pos

if __name__ == '__main__':
    test = MyDataset(data_path="../data/ex.csv")
    print(test.__getitem__(index=8)[0].shape)
    print(test.__getitem__(index=8)[1].shape)
