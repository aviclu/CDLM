import numpy as np
import collections
import torch



class Corpus:
    def __init__(self, documents, tokenizer, segment_window, mentions, subtopic=True, predicted_topics=None):
        self.documents = documents
        self.mentions = mentions
        self.segment_window = segment_window

        self.topic_list = []
        self.topics_list_of_docs = []
        self.topics_origin_tokens = []
        self.topics_bert_tokens = []
        self.topics_start_end_bert = []

        self.labels = self.create_dict_labels()

        if predicted_topics:
            self.docs_by_topic = self.separate_doc_into_predicted_subtopics(predicted_topics)
        else:
            self.docs_by_topic = self.separate_docs_into_topics(subtopic)

        self.tokenize(tokenizer)



    def create_dict_labels(self):
        label_dict = collections.defaultdict(dict)
        for m in self.mentions:
            label_dict[m['doc_id']][(min(m['tokens_ids']), max(m['tokens_ids']))] = m['cluster_id']

        return label_dict



    def get_candidate_labels(self, doc_ids, starts, ends):
        labels = [0] * len(doc_ids)
        starts = starts.tolist()
        ends = ends.tolist()
        for i, (doc_id, start, end) in enumerate(zip(doc_ids, starts, ends)):
            if doc_id in self.labels:
                label = self.labels[doc_id].get((start, end), None)
                if label:
                    labels[i] = label

        return torch.tensor(labels)



    def separate_doc_into_predicted_subtopics(self, predicted_subtopics):
        '''
        Function to init the predicted subtopics as Shany Barhom
        :param predicted_subtopics: Shany's file
        :return:
        '''
        text_by_subtopics = collections.defaultdict(list)
        for i, doc_list in enumerate(predicted_subtopics):
            for doc in doc_list:
                doc_key = doc #+ '.xml'
                if doc_key in self.documents:
                    text_by_subtopics[i].append(doc_key)

        return text_by_subtopics



    def separate_docs_into_topics(self, subtopic):
        docs_by_topics = collections.defaultdict(list)
        for doc_id, tokens in self.documents.items():
            topic_key = doc_id.split('_')[0]
            if subtopic:
                topic_key += '_{}'.format(1 if 'plus' in doc_id else 0)
            docs_by_topics[topic_key].append(doc_id)

        return docs_by_topics



    def split_doc_into_segments(self, bert_tokens, sentence_ids, with_special_tokens=True):
        segments = [0]
        current_token = 0
        max_segment_length = self.segment_window
        if with_special_tokens:
            max_segment_length -= 2
        while current_token < len(bert_tokens):
            end_token = min(len(bert_tokens) - 1, current_token + max_segment_length - 1)
            sentence_end = sentence_ids[end_token]
            if end_token != len(bert_tokens) - 1 and sentence_ids[end_token + 1] == sentence_end:
                while end_token >= current_token and sentence_ids[end_token] == sentence_end:
                    end_token -= 1

                if end_token < current_token:
                    raise ValueError(bert_tokens)

            current_token = end_token + 1
            segments.append(current_token)

        return segments



    def tokenize_topic(self, topic, tokenizer):
        list_of_docs = []
        docs_bert_tokens = []
        docs_origin_tokens = []
        docs_start_end_bert = []

        for doc_id in self.docs_by_topic[topic]:
            tokens = self.documents[doc_id]
            bert_tokens_ids, bert_sentence_ids = [], []
            start_bert_idx, end_bert_idx = [], []
            original_tokens = []
            alignment = []
            bert_cursor = -1

            for i, token in enumerate(tokens):
                sent_id, token_id, token_text, flag_sentence = token
                bert_token = tokenizer.encode(token_text, add_special_tokens=True)[1:-1]

                if bert_token:
                    bert_tokens_ids.extend(bert_token)
                    bert_start_index = bert_cursor + 1
                    start_bert_idx.append(bert_start_index)
                    bert_cursor += len(bert_token)
                    bert_end_index = bert_cursor
                    end_bert_idx.append(bert_end_index)
                    original_tokens.append([sent_id, token_id, token_text, flag_sentence])
                    bert_sentence_ids.extend([sent_id] * len(bert_token))
                    alignment.extend([token_id] * len(bert_token))


            segments = self.split_doc_into_segments(bert_tokens_ids, bert_sentence_ids)
            ids = [x[1] for x in original_tokens]
            bert_segments, original_segments, start_end_segment = [], [], []
            delta = 0

            for start, end in zip(segments, segments[1:]):
                original_start = ids.index(alignment[start])
                original_end = ids.index(alignment[end - 1])

                bert_start = np.array(start_bert_idx[original_start:original_end + 1]) - delta
                bert_end = np.array(end_bert_idx[original_start:original_end + 1]) - delta

                original_segments.append(original_tokens[original_start:original_end + 1])
                bert_ids = tokenizer.encode(' '.join([x[2] for x in original_tokens[original_start:original_end + 1]]),
                                            add_special_tokens=True)[1:-1]

                if len(bert_ids) != (end - start):
                    raise Exception(doc_id, start, end, len(bert_ids), (end - start))

                bert_segments.append(bert_ids)
                start_end = np.concatenate((np.expand_dims(bert_start, 1),
                                            np.expand_dims(bert_end, 1)), axis=1)
                start_end_segment.append(start_end)
                delta = end

            segment_doc = [doc_id] * (len(segments) - 1)
            docs_start_end_bert.extend(start_end_segment)
            list_of_docs.extend(segment_doc)
            docs_bert_tokens.extend(bert_segments)
            docs_origin_tokens.extend(original_segments)

        return list_of_docs, docs_origin_tokens, docs_bert_tokens, docs_start_end_bert



    def tokenize(self, tokenizer):
        for topic in self.docs_by_topic:
            list_of_docs, docs_origin_tokens, docs_bert_tokens, docs_start_end_bert = \
            self.tokenize_topic(topic, tokenizer)

            self.topic_list.append(topic)
            self.topics_list_of_docs.append(list_of_docs)
            self.topics_origin_tokens.append(docs_origin_tokens)
            self.topics_bert_tokens.append(docs_bert_tokens)
            self.topics_start_end_bert.append(docs_start_end_bert)