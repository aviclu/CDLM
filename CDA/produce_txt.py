import csv
from nltk.tokenize import sent_tokenize, word_tokenize
goal = 'test'

data_path = '{}_ai2_gorc.csv'.format(goal)

def process(text):
        text_array = []
        for sentence in sent_tokenize(text=text):
            text_array.append(sentence)
        return text_array

f = open('_{}_ai2_gorc.txt'.format(goal), 'w')
g = open('_{}_ai2_gorc.index'.format(goal), 'w')
with open(data_path) as csv_file:
    reader = csv.reader(csv_file, quotechar='"')
    for idx, line in enumerate(reader):
        text_1 = ""
        text_2 = ""
        line[1] = line[1].replace('\001', '')
        line[2] = line[2].replace('\001', '')
        label = int(line[0])
        text_1 = process(text_1)
        text_2 = process(text_2)
        idx_1 = len(text_1)
        idx_2 = len(text_2)
        g.write('{}\n'.format(idx_1))
        g.write('{}\n'.format(idx_2))
        for i in text_1:
            f.write(i+'\n')
        for i in text_2:
            f.write(i+'\n')
        
