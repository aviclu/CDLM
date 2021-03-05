import logging
import os
import pickle
import time
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import random

from transformers import torch_distributed_zero_first
from transformers import PreTrainedTokenizer, AutoModelWithLMHead

logger = logging.getLogger(__name__)

def read_in_train_set(input_path):
	corpus = []
	with open(input_path, 'r', encoding="utf8") as fr:
		for line in fr:
			corpus.append(line.strip())
	return corpus




pronouns_file = open('pronouns')
pronouns = pronouns_file.readlines()
pronouns = [x[:-1] for x in pronouns]
pronouns += ['s','’','ll','d']
Pronoun = ['another', 'anybody', 'anymore', 'anyone', 'anything', 'deez', 'everybody', 'everyday', 'everyone',
               'everything', 'he', "he'd", "he's", 'her', 'hers', 'herself', 'hes', 'him', 'himself', 'his', 'i', "i'd",
               "i'd've", "i'll", "i'm", "i've", 'id', 'idc', 'idgaf', 'idk', 'idontknow', 'idve', 'ikr', 'ilya', 'im',
               'ima', 'imean', 'imma', 'it', "it'd", "it'll", "it's", 'itd', 'itll', 'its', 'itself', 'ive', "let's",
               'lets', 'me', 'methinks', 'mine', 'my', 'myself', 'nobody', 'oneself', 'other', 'others', 'our', 'ours',
               'ourselves', 'she', "she'd", "she'll", "she's", 'shes', 'somebody', 'someone', 'something',
               'somethingness', 'somewhere', 'stuff', 'that', "that'd", "that'll", "that's", 'thatd', 'thatll', 'thats',
               'thee', 'their', 'theirs', 'theirselves', 'them', 'themself', 'themselves', 'these', 'they', "they'd",
               "they'll", "they've", 'theyd', 'theyll', 'theyve', 'thine', 'thing', 'thingy', 'thingamabob',
               'thingness', 'this', 'those', 'thou', 'thoust', 'thy', 'thyself', 'u', 'ur', 'us', 'we', "we'd", "we'll",
               "we're", "we've", 'weve', 'what', "what'd", "what'll", "what's", 'whatd', 'whatever', 'whatll', 'whats',
               'which', 'whichever', 'who', "who'd", "who'll", "who's", 'whod', 'whoever', 'wholl', 'whom', 'whomever',
               'whos', 'whose', 'whosever', "y'all", "y'all's", 'ya', 'yall', 'yalls', 'ye', 'you', "you'd", "you'll",
               "you're", "you've", 'youd', 'youll', 'your', 'youre', 'yours', 'yourself', 'yourselves', 'youve']
Ppron = ['he', "he'd", "he's", 'her', 'hers', 'herself', 'hes', 'him', 'himself', 'his', 'i', "i'd", "i'd've",
         "i'll", "i'm", "i've", 'id', 'idc', 'idgaf', 'idk', 'idontknow', 'idve', 'ikr', 'ilya', 'im', 'ima',
         'imean', 'imma', 'ive', "let's", 'lets', 'me', 'methinks', 'mine', 'my', 'myself', 'oneself', 'our',
         'ours', 'ourselves', 'she', "she'd", "she'll", "she's", 'shes', 'thee', 'their', 'theirs', 'theirselves',
         'them', 'themself', 'themselves', 'they', "they'd", "they'll", "they've", 'theyd', 'theyll', 'theyve',
         'thine', 'thou', 'thoust', 'thy', 'thyself', 'u', 'ur', 'us', 'we', "we'd", "we'll", "we're", "we've",
         'weve', "y'all", "y'all's", 'ya', 'yall', 'yalls', 'ye', 'you', "you'd", "you'll", "you're", "you've",
         'youd', 'youll', 'your', 'youre', 'yours', 'yourself', 'yourselves', 'youve', 'Ppron']
I = ['i', "i'd", "i'd've", "i'll", "i'm", "i've", 'id', 'idc', 'idgaf', 'idk', 'idontknow', 'idve', 'ikr', 'ilya',
     'im', 'ima', 'imean', 'imma', 'ive', 'me', 'methinks', 'mine', 'my', 'myself']
We = ["let's", 'lets', 'our', 'ours', 'ourselves', 'us', 'we', "we'd", "we'll", "we're", "we've", 'weve']
You = ['ilya', 'thee', 'thine', 'thou', 'thoust', 'thy', 'thyself', 'u', 'ur', "y'all", "y'all's", 'ya', 'yall',
       'yalls', 'ye', 'you', "you'd", "you'll", "you're", "you've", 'youd', 'youll', 'your', 'youre', 'yours',
       'yourself', 'yourselves', 'youve', 'You']
SheHe = ['he', "he'd", "he's", 'her', 'hers', 'herself', 'hes', 'him', 'himself', 'his', 'oneself', 'she', "she'd",
         "she'll", "she's", 'shes']
They = ['their', 'theirs', 'theirselves', 'them', 'themself', 'themselves', 'they', "they'd", "they'll", "they've",
        'theyd', 'theyll', 'theyve']
Ipron = ['another', 'anybody', 'anymore', 'anyone', 'anything', 'deez', 'everybody', 'everyday', 'everyone',
             'everything', 'it', "it'd", "it'll", "it's", 'itd', 'itll', 'its', 'itself', 'nobody', 'other', 'others',
             'somebody', 'someone', 'something', 'somethingness', 'stuff', 'that', "that'd", "that'll", "that's",
             'thatd', 'thatll', 'thats', 'these', 'thing', 'thingy', 'thingamabob', 'thingness', 'this', 'those',
             'what', "what'd", "what'll", "what's", 'whatd', 'whatever', 'whatll', 'whats', 'which', 'whichever', 'who',
             "who'd", "who'll", "who's", 'whod', 'whoever', 'wholl', 'whom', 'whomever', 'whos', 'whose', 'whosever']
pronouns += Ppron + I + We + You + SheHe + They + Ipron
pronouns = set(pronouns)
def words_to_token_ids(sent, tokenizer,stard_id=0):
    idx = stard_id
    enc = [tokenizer.encode(x, add_special_tokens=False, add_prefix_space=False) for x in sent.split()]
    desired_output = []
    for token in enc:
        tokenoutput = []
        for ids in token:
            tokenoutput.append(idx)
            idx += 1
        desired_output.append(tokenoutput)
    return desired_output

def select_words_to_mask_special_tokens_only_multiple_docs(sentenses, tokenizer,block_size):
    curr_len = 0
    i = 0
    attention_mask = np.array([0] * block_size)
    stops = {'...', '.', '?', '!', '!!!'}
    words = []
    while curr_len < block_size and i < len(sentenses):
        sent = sentenses[i]
        words1 = sent.split()
        new_words1 = []
        new_words1.append('<doc-s>')
        new_words1.append(tokenizer.bos_token)
        for k in range(len(words1)):
            x = words1[k]
            new_words1.append(x)
            if x in stops:
                new_words1.append(tokenizer.eos_token)
                if k == len(words1)-1:
                    new_words1.append(tokenizer.eos_token)
                else:
                    new_words1.append(tokenizer.bos_token)
        new_words1.append('</doc-s>')
        bert_toks = [tokenizer.encode(x, add_special_tokens=False, add_prefix_space=False) for x in new_words1]
        bert_toks = [item for sublist in bert_toks for item in sublist]
        if curr_len + len(bert_toks) > block_size:
            break
        words += new_words1
        i += 1
        curr_len += len(bert_toks)
    words.append('</doc-s>')
    examp = [tokenizer.encode(x, add_special_tokens=False, add_prefix_space=False) for x in words]
    examp = [item for sublist in examp for item in sublist]
    stat = len(examp)
    new_words = ' '.join(words)
    words_list = words_to_token_ids(new_words, tokenizer)
    blacklist1 = [i for i in range(len(words)) if words[i] in pronouns]
    toks_to_remove = [words_list[x] for x in blacklist1]
    toks_to_remove = [x for sub in toks_to_remove for x in sub]

    toks_to_remove = set(toks_to_remove)
    to_pad_length = block_size - len(examp)
    examp += to_pad_length * [tokenizer.pad_token_id]
    attention_mask[list(toks_to_remove)] = 1
    attention_mask[-to_pad_length:] = 1
    return (examp, attention_mask), stat


class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
        self,  tokenizer: PreTrainedTokenizer ,split:str, file_path: str, block_size: int, overwrite_cache=False, local_rank=-1
    ):
        block_size = 4096
        self.block_size = 4096

        directory = './processed_files'
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), split,),
        )

        with torch_distributed_zero_first(local_rank):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                self.examples = [ex for ex in self.examples[:int(len(self.examples))] if len(ex[0])==self.block_size and len(ex[1])==self.block_size]
                self.examples = [ex for ex in self.examples[:]]
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                input_path = './multinews/train.txt.src'
                logger.info(f"Creating features from dataset file at {input_path}")
                encoder_func = select_words_to_mask_special_tokens_only_multiple_docs
                self.examples = []
                self.masking_samples = []
                corpus = read_in_train_set(input_path)
                ln = []
                for i in range(len(corpus)):
                    sample = corpus[i].strip()
                    articles = sample.split("story_separator_special_tag")
                    ln.append(articles)
                tokenizer.add_tokens(['<doc-s>'], special_tokens=True)
                tokenizer.add_tokens(['</doc-s>'], special_tokens=True)
                stats = []
                while len(stats) < 64*25*1000:
                    for topic in ln:
                        if len(topic) > 2:
                            s = random.sample(topic, len(topic))
                            examp, st = encoder_func(s, tokenizer, block_size)
                            self.examples.append(examp)
                            stats.append(st)
                # Uncomment for creating data for the random baseline
                # while len(self.examples) < 64*25*1000:
                #     s = random.sample(ln, 10)
                #     curr_false_topic = []
                #     for topic in s:
                #         curr_false_topic.append(random.sample(topic, 1)[0])
                #     examp, st = encoder_func(curr_false_topic, tokenizer, block_size)
                #     self.examples.append(examp)
                #     stats.append(st)
                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    f"Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)

