import pandas as pd
from torch.utils.data.dataset import Dataset
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import re


class MyDataset(Dataset):

    def __init__(self, data_path, word_to_idx, max_length_sentences=30, max_length_word=35):
        super(MyDataset, self).__init__()

        texts, labels = [], []
        with open(data_path) as csv_file:
            reader = csv.reader(csv_file, quotechar='"')
            for idx, line in enumerate(reader):
                text_1 = ""
                text_2 = ""
                for tx in line[1].split():
                    text_1 += tx.lower()
                    text_1 += " "
                for tx in line[2].split():
                    text_2 += tx.lower()
                    text_2 += " "
                label = int(line[0])
                texts.append((text_1, text_2))
                labels.append(label)
                '''
                if label==0:
                    labels.append(-1)
                else:
                    labels.append(1)
                '''
        self.texts = texts
        self.labels = labels
        #self.dict = [word[0] for word in self.dict]
        self.dict_index = word_to_idx
        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word
        self.num_classes = len(set(self.labels))

    def __len__(self):
        return len(self.labels)
    
    def sent_tokenize_cn(self, para):
        para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
        para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        return para.split("\n")

    def process(self, text):
        '''
        document_encode = [
            [self.dict.index(word) if word in self.dict else -1 for word in word_tokenize(text=sentences)] for sentences
            in
            sent_tokenize(text=text)]
        '''
        document_encode = [
            [self.dict_index.get(word,-1) for word in sentences.split()] for sentences
            in self.sent_tokenize_cn(para=text)]
        #print(document_encode)
        
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
        return document_encode


    def __getitem__(self, index):
        label = self.labels[index]
        text_1 = self.texts[index][0]
        text_2 = self.texts[index][1]
        text_1 = self.process(text_1)
        text_2 = self.process(text_2)
        return text_1.astype(np.int64), text_2.astype(np.int64) ,label

if __name__ == '__main__':
    test = MyDataset(data_path="../data/test_pair.csv", dict_path="../data/glove.6B.50d.txt")
    print(test.__getitem__(index=1)[0].shape)
    print(test.__getitem__(index=1)[1].shape)
    
