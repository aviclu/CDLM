from torch.utils.data.dataset import Dataset
import torch
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import sys
import os
from transformers import *
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

class MyDataset(Dataset):
    def __init__(self, data_path, max_len=18):
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
        bd_pairs = []
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
                line[1] = line[1].replace('\001', '') 
                for index, tx in enumerate(sent_tokenize(line[1])):
                    if '\001' in tx:
                        cite_pos.append(index)
                pos.append(cite_pos)
        #texts = self.unifying(texts, bd_list)
        labels = np.array(labels)
        self.texts = bd_pairs
        self.labels = labels
        self.pos = pos

    def get_pos(self):
        return self.pos

    def __len__(self):
        return len(self.labels)

    def process(self, bd):
        i = self.sentences[bd[0]:bd[1]]
        i = np.array(i)
        max_length = self.max_len
        if i.shape[0] < max_length:
            padding = np.zeros((max_length-i.shape[0],1024))
            doc = np.concatenate((i, padding))
        else:
            doc = i[:max_length]
        return doc
        
    def __getitem__(self, index):
        cite_pos = self.pos[index]
        label = self.labels[index]
        text_1 = self.process(self.texts[index][0])
        text_2 = self.process(self.texts[index][1])
        #cls_ = self.process(text_1, text_2)
        cls1 = text_1
        cls2 = text_2
        return cls1.astype(np.float32), cls2.astype(np.float32), label, cite_pos

if __name__ == '__main__':
    test = MyDataset(data_path="../data/ex.csv")
    print(test.__getitem__(index=8)[0].shape)
    print(test.__getitem__(index=8)[1].shape)
