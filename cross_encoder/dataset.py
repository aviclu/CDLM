import torch
from torch.utils import data
import collections
from itertools import combinations
import numpy as np
import os
import json
import copy
import transformers
import pickle

class CrossEncoderDataset(data.Dataset):
    def __init__(self, mentions_repr, first, second, labels):
        self.instances = [' '.join(['<g>',"<doc-s>", mentions_repr[first[i]], "</doc-s>","<doc-s>",
                                    mentions_repr[second[i]], "</doc-s>"]) for i in range(len(first))]
        self.labels = labels.to(torch.float)


    def __len__(self):
        return len(self.instances)


    def __getitem__(self, index):
        return self.instances[index], self.labels[index].unsqueeze(-1)




class CrossEncoderDatasetInstances(data.Dataset):
    def __init__(self, instances):
        self.instances = instances


    def __len__(self):
        return len(self.instances)


    def __getitem__(self, index):
        return self.instances[index]



class CrossEncoderDatasetFull(data.Dataset):
    def __init__(self, config, split_name, same_lemma=False, partial=False, mode=None):
        self.mode = mode
        with open('predicted_topics', 'rb') as handle:
            self.predicted_topics = pickle.load(handle)
            self.predicted_topics = {x + '.xml':i for i, lst in enumerate(self.predicted_topics) for x in lst}
        self.read_files(config, split_name)
        self.lemmas = np.asarray([x['lemmas'] for x in self.mentions])
        self.topics = set([m['topic'] for m in self.mentions])
        self.mention_labels = torch.tensor([m['cluster_id'] for m in self.mentions])
        self.doc_dict, self.doc_comp, self.doc_comp_no_seps = self.make_dict_of_sentences(self.documents)
        self.doc_tokens_word2id, self.doc_tokens = self.compute_tokenization()
        if split_name == 'train' or split_name == 'dev':
            self.mentions_by_topics = collections.defaultdict(list)
            for i, m in enumerate(self.mentions):
                self.mentions_by_topics[m['topic']].append(i)
        else:
            self.mentions_by_topics = collections.defaultdict(list)
            for i, m in enumerate(self.mentions):
                self.mentions_by_topics[m['pred_topic']].append(i)

        self.first = []
        self.second = []
        self.labels = []

        for topic, mentions in self.mentions_by_topics.items():
            first, second = zip(*list(combinations(range(len(mentions)), 2)))
            mentions = torch.tensor(mentions)
            first, second = torch.tensor(first), torch.tensor(second)
            first, second = mentions[first], mentions[second]
            labels = (self.mention_labels[first] != 0) & (self.mention_labels[second] != 0) \
                     & (self.mention_labels[first] == self.mention_labels[second])

            self.first.extend(first)
            self.second.extend(second)
            self.labels.extend(labels)

        self.first = torch.tensor(self.first)
        self.second = torch.tensor(self.second)
        self.labels = torch.tensor(self.labels, dtype=torch.float)


        if same_lemma:
            idx = (self.lemmas[self.first] == self.lemmas[self.second]).nonzero()
            self.first = self.first[idx]
            self.second = self.second[idx]
            self.labels = self.labels[idx]



        self.instances = self.prepare_pair_of_mentions(self.mentions, self.first,
                                                       self.second)
        if partial:
            self.instances = self.instances[:int(0.3*len(self.labels))]
            self.labels = self.labels[:len(self.instances)]


    def read_files(self, config, split_name):
        docs_path = os.path.join(config.data_folder, split_name + '.json')
        mentions_path = os.path.join(config.data_folder,
                                     split_name + '_{}.json'.format(config.mention_type))
        with open(docs_path, 'r') as f:
            self.documents = json.load(f)

        self.mentions = []
        if config.use_gold_mentions:
            with open(mentions_path, 'r') as f:
                self.mentions = json.load(f)
            if split_name == 'test':
                for x in self.mentions:
                    x['pred_topic'] = self.predicted_topics[x['doc_id']]



    def make_dict_of_sentences(self, documents):
        doc_dict = {}
        doc_compp = {doc:[] for doc, _ in documents.items()}
        for doc, tokens in documents.items():
            dict = collections.defaultdict(list)
            for i, (sentence_id, token_id, text, flag) in enumerate(tokens):
                dict[sentence_id].append([token_id, sentence_id, text, flag])
                doc_compp[doc].append([token_id, sentence_id, text, flag])
            doc_dict[doc] = dict
        doc_comp = copy.deepcopy(doc_dict)
        for k, v in doc_comp.items():
            for s,j  in v.items():
                j.insert(0, [np.inf, s, "<s>", j[0][-1]])
                j.insert(len(j), [np.inf, s, "<\s>", j[0][-1]])
        doc_comp_new = {doc:[] for doc, _ in documents.items()}
        for k, v in doc_comp.items():
            for s,j in v.items():
                doc_comp_new[k].extend(j)
        return doc_dict, doc_comp_new, doc_compp

    def encode_mention_with_context(self, mention):
        doc_id, sentence_id = mention['doc_id'], int(mention['sentence_id'])
        tokens = self.doc_dict[doc_id][sentence_id]
        token_ids = [x[0] for x in tokens]

        start_idx = token_ids.index(min(mention['tokens_ids']))
        end_idx = token_ids.index(max(mention['tokens_ids'])) + 1

        mention_repr = [x[2] for x in tokens[:start_idx]] + ["<m>"] \
                       + [x[2] for x in tokens[start_idx:end_idx]] + ["</m>"] \
                       + [x[2] for x in tokens[end_idx:]]

        return ' '.join(mention_repr)

    def words_to_token_ids(self, tokens, tokenizer, stard_id=0):
        idx = stard_id
        enc = [tokenizer.encode(x, add_special_tokens=False, add_prefix_space=False) for x in tokens]
        desired_output = []
        for token in enc:
            tokenoutput = []
            for ids in token:
                tokenoutput.append(idx)
                idx += 1
            desired_output.append(tokenoutput)
        enc = [x for s in enc for x in s]
        a = [i for i, x in enumerate(desired_output) for s in x]
        return desired_output, enc, a

    def compute_tokenization(self):
        d = {}
        d2 = {}
        tokenizer = transformers.LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        tokenizer_rob = transformers.RobertaTokenizer.from_pretrained('roberta-base')

        for k,v in self.doc_comp.items():
            toks = [x[2] for x in v]
            doc = ' '.join(toks)
            desired_output, enc , a= self.words_to_token_ids(toks, tokenizer)
            d[k] = desired_output
            d2[k] = a
        return d, d2

    def encode_mention_with_context_long(self, mention):
        doc_id, sentence_id = mention['doc_id'], int(mention['sentence_id'])
        tokens = self.doc_comp[doc_id]
        token_ids = [x[0] for x in tokens]

        start_idx = token_ids.index(min(mention['tokens_ids']))
        end_idx = token_ids.index(max(mention['tokens_ids'])) + 1
        doc_tokens_start = len([s for x in self.doc_tokens_word2id[doc_id][:start_idx] for s in x])
        doc_tokens_end = len([s for x in self.doc_tokens_word2id[doc_id][:end_idx] for s in x])
        len_doc_tokens = len(self.doc_tokens[doc_id])
        tokens_limit = 2048-4
        if len_doc_tokens > tokens_limit:
            id_to_trun = self.doc_tokens[doc_id][len_doc_tokens - tokens_limit]
            if doc_tokens_start > int(len_doc_tokens*0.5):
                tokens = tokens[id_to_trun:]
            else:
                tokens = tokens[:-id_to_trun]
            token_ids = [x[0] for x in tokens]
            start_idx = token_ids.index(min(mention['tokens_ids']))
            end_idx = token_ids.index(max(mention['tokens_ids'])) + 1
        mention_repr = [x[2] for x in tokens[:start_idx]] + ["<m>"] \
                       + [x[2] for x in tokens[start_idx:end_idx]] + ["</m>"] \
                       + [x[2] for x in tokens[end_idx:]]

        return ' '.join(mention_repr)

    def encode_mention_with_context_long_no_sep(self, mention):
        doc_id, sentence_id = mention['doc_id'], int(mention['sentence_id'])
        tokens = self.doc_comp_no_seps[doc_id]
        token_ids = [x[0] for x in tokens]

        start_idx = token_ids.index(min(mention['tokens_ids']))
        end_idx = token_ids.index(max(mention['tokens_ids'])) + 1
        doc_tokens_start = len([s for x in self.doc_tokens_word2id[doc_id][:start_idx] for s in x])
        doc_tokens_end = len([s for x in self.doc_tokens_word2id[doc_id][:end_idx] for s in x])
        len_doc_tokens = len(self.doc_tokens[doc_id])
        tokens_limit = 2048-4
        if len_doc_tokens > tokens_limit:
            id_to_trun = self.doc_tokens[doc_id][len_doc_tokens - tokens_limit]
            if doc_tokens_start > int(len_doc_tokens*0.5):
                tokens = tokens[id_to_trun:]
            else:
                tokens = tokens[:-id_to_trun]
            token_ids = [x[0] for x in tokens]
            start_idx = token_ids.index(min(mention['tokens_ids']))
            end_idx = token_ids.index(max(mention['tokens_ids'])) + 1

        mention_repr = [x[2] for x in tokens[:start_idx]] + ["<m>"] \
                       + [x[2] for x in tokens[start_idx:end_idx]] + ["</m>"] \
                       + [x[2] for x in tokens[end_idx:]]

        return ' '.join(mention_repr)

    def encode_mention_with_context_parag(self, mention):
        doc_id, sentence_id = mention['doc_id'], int(mention['sentence_id'])
        tokens = self.doc_comp_no_seps[doc_id]
        token_ids = [x[0] for x in tokens]

        start_idx = token_ids.index(min(mention['tokens_ids']))
        end_idx = token_ids.index(max(mention['tokens_ids'])) + 1
        doc_tokens_start = len([s for x in self.doc_tokens_word2id[doc_id][:start_idx] for s in x])
        doc_tokens_end = len([s for x in self.doc_tokens_word2id[doc_id][:end_idx] for s in x])
        len_doc_tokens = len(self.doc_tokens[doc_id])
        tokens_limit = 2048-4
        if len_doc_tokens > tokens_limit:
            id_to_trun = self.doc_tokens[doc_id][len_doc_tokens - tokens_limit]
            if doc_tokens_start > int(len_doc_tokens*0.5):
                tokens = tokens[id_to_trun:]
            else:
                tokens = tokens[:-id_to_trun]
            token_ids = [x[0] for x in tokens]
            start_idx = token_ids.index(min(mention['tokens_ids']))
            end_idx = token_ids.index(max(mention['tokens_ids'])) + 1

        mention_repr = [x[2] for x in tokens[:start_idx]] + ["<m>"] \
                       + [x[2] for x in tokens[start_idx:end_idx]] + ["</m>"] \
                       + [x[2] for x in tokens[end_idx:]]

        return ' '.join(mention_repr)

    def prepare_mention_representation(self, mentions):
        if self.mode is None:
            return np.asarray([self.encode_mention_with_context_long(m) for m in mentions])
        if self.mode == 'cdmlm':
            return np.asarray([self.encode_mention_with_context_long(m) for m in mentions])
        if self.mode == 'long':
            return np.asarray([self.encode_mention_with_context_long_no_sep(m) for m in mentions])
        if self.mode == 'reg':
            return np.asarray([self.encode_mention_with_context(m) for m in mentions])


    def prepare_pair_of_mentions(self, mentions, first, second):
        if self.mode is None:
            mentions_repr = np.asarray([self.encode_mention_with_context_long(m) for m in mentions])
            instances = [' '.join(['<g>', "<doc-s>", mentions_repr[first[i]], "</doc-s>","<doc-s>",
                                        mentions_repr[second[i]], "</doc-s>"]) for i in range(len(first))]

            return instances
        if self.mode == 'cdmlm':
            mentions_repr = np.asarray([self.encode_mention_with_context_long(m) for m in mentions])
            instances = [' '.join(['<g>', "<doc-s>", mentions_repr[first[i]], "</doc-s>","<doc-s>",
                                        mentions_repr[second[i]], "</doc-s>"]) for i in range(len(first))]
            return instances
        if self.mode == 'long':
            mentions_repr = np.asarray([self.encode_mention_with_context_long_no_sep(m) for m in mentions])
            instances = [' '.join([mentions_repr[first[i]], "<s>",
                                        mentions_repr[second[i]], "</s>"]) for i in range(len(first))]
            return instances
        if self.mode == 'reg':
            mentions_repr = np.asarray([self.encode_mention_with_context(m) for m in mentions])
            instances = [' '.join([mentions_repr[first[i]], "<s>",
                                        mentions_repr[second[i]], "</s>"]) for i in range(len(first))]
            return instances



    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):
        return self.instances[index], self.labels[index].unsqueeze(-1)





class CrossEncoderDatasetTopic(data.Dataset):
    def __init__(self, full_dataset, topic):
        super(CrossEncoderDatasetTopic, self).__init__()
        self.topic_mentions_ids = full_dataset.mentions_by_topics[topic]
        self.topic_mentions = [full_dataset.mentions[x] for x
                               in self.topic_mentions_ids]

        first, second = zip(*list(combinations(range(len(self.topic_mentions)), 2)))
        self.first, self.second = torch.tensor(first), torch.tensor(second)

        self.instances = full_dataset.prepare_pair_of_mentions(
            self.topic_mentions, self.first, self.second)



    def __len__(self):
        return len(self.topic_mentions)


    def __getitem__(self, index):
        return self.instances[index]

