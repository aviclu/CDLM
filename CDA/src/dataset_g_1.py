import pandas as pd
from torch.utils.data.dataset import Dataset
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import sys 
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

    def __init__(self, data_path, dict_path, max_length_sentences=30, max_length_word=35):
        super(MyDataset, self).__init__()

        texts, labels, pos = [], [], []
        with open(data_path) as csv_file:
            reader = csv.reader(csv_file, quotechar='"')
            for idx, line in enumerate(reader):
                text_1 = []
                text_2 = []
                cite_pos = []
                for index, tx in enumerate(sent_tokenize(line[1])):
                    if '\001' in tx:
                        cite_pos.append(index)
                    text_1.append(tx.lower())
                for tx in sent_tokenize(line[2]):
                    text_2.append(tx.lower())
                label = int(line[0])
                texts.append((text_1, text_2))
                labels.append(label)
                pos.append(cite_pos)
        self.texts = texts
        self.labels = labels
        self.pos = pos
        self.dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
                                usecols=[0]).values
        self.dict = [word[0] for word in self.dict]
        #self.dict = [word[0] for word in self.dict]
        self.dict_index = {}
        for index, word in enumerate(self.dict):
            self.dict_index[word] = index
        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word
        self.num_classes = len(set(self.labels))
        self.mask = []

    def __len__(self):
        return len(self.labels)

    def process(self, text):
        '''
        document_encode = [
            [self.dict.index(word) if word in self.dict else -1 for word in word_tokenize(text=sentences)] for sentences
            in
            sent_tokenize(text=text)]
        '''
        document_encode = [
            [self.dict_index.get(word,-1) for word in word_tokenize(text=sentences)] for sentences in text]

        if len(document_encode) < self.max_length_sentences:            
            mask = [1]*len(document_encode)
            mask += [0]*self.max_length_sentences
            mask = mask[:self.max_length_sentences]
        else:
            mask = [1]*self.max_length_sentences 

        for sentences in document_encode:
            if len(sentences) < self.max_length_word:
                extended_words = [-1 for _ in range(self.max_length_word - len(sentences))]
                sentences.extend(extended_words)

        if len(document_encode) < self.max_length_sentences:
            extended_sentences = [[-1 for _ in range(self.max_length_word)] for _ in
                                  range(self.max_length_sentences - len(document_encode))]
            document_encode.extend(extended_sentences)

        document_encode = [sentences[:self.max_length_word] for sentences in document_encode][
                          :self.max_length_sentences]

        document_encode = np.stack(arrays=document_encode, axis=0)
        document_encode += 1
        self.collect_mask(np.array(mask))
        return document_encode

    def get_pos(self):
        return self.pos
    def collect_mask(self, mask):
        self.mask.append(mask)

    def get_mask(self):
        return np.array(self.mask)

    def __getitem__(self, index):
        cite_pos = self.pos[index]
        label = self.labels[index]
        text_1 = self.texts[index][0]
        text_2 = self.texts[index][1]
        text_1 = self.process(text_1)
        text_2 = self.process(text_2)
        return text_1.astype(np.int64), text_2.astype(np.int64) ,label, cite_pos

if __name__ == '__main__':
    test = MyDataset(data_path="../data/test_pair.csv", dict_path="../data/glove.6B.50d.txt")
    print(test.__getitem__(index=1)[0].shape)
    print(test.__getitem__(index=1)[1].shape)
    
