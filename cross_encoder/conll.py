import collections
import operator
import os



def get_dict_map(predictions, doc_ids, starts, ends):
  doc_start_map = collections.defaultdict(list)
  doc_end_map = collections.defaultdict(list)
  doc_word_map = collections.defaultdict(list)

  for cluster_id, mentions in predictions.items():
      for idx in mentions:
          doc_id, start, end = doc_ids[idx], starts[idx], ends[idx]
          start_key = doc_id + '_' + str(start)
          end_key = doc_id + '_' + str(end)
          if start == end:
              doc_word_map[start_key].append(cluster_id)
          else:
              doc_start_map[start_key].append((cluster_id, end))
              doc_end_map[end_key].append((cluster_id, start))

  for k, v in doc_start_map.items():
      doc_start_map[k] = [cluster_id for cluster_id, end_key in sorted(v, key=operator.itemgetter(1), reverse=True)]
  for k, v in doc_end_map.items():
      doc_end_map[k] = [cluster_id for cluster_id, end_key in sorted(v, key=operator.itemgetter(1), reverse=True)]

  return doc_start_map, doc_end_map, doc_word_map



def output_conll(data, doc_word_map, doc_start_map, doc_end_map):
    predicted_conll = []
    for doc_id, tokens in data.items():
        topic = doc_id.split('_')[0]
        subtopic = topic + '_{}'.format(1 if 'plus' in doc_id else 0)
        for sentence_id, token_id, token_text, flag in tokens:
            if not flag:
                continue
            clusters = '-'
            coref_list = list()
            if flag:
                token_key = doc_id + '_' + str(token_id)
                if token_key in doc_word_map:
                    for cluster_id in doc_word_map[token_key]:
                        coref_list.append('({})'.format(cluster_id))
                if token_key in doc_start_map:
                    for cluster_id in doc_start_map[token_key]:
                        coref_list.append('({}'.format(cluster_id))
                if token_key in doc_end_map:
                    for cluster_id in doc_end_map[token_key]:
                        coref_list.append('{})'.format(cluster_id))

            if len(coref_list) > 0:
                clusters = '|'.join(coref_list)

            predicted_conll.append([topic, subtopic, doc_id, sentence_id, token_id, token_text, flag, clusters])


    return predicted_conll




def write_output_file(data, predictions, doc_ids, starts, ends, dir_path, doc_name, topic_level=True, corpus_level=True):
    doc_start_map, doc_end_map, doc_word_map = get_dict_map(predictions, doc_ids, starts, ends)
    corpus_level_tokens = output_conll(data, doc_word_map, doc_start_map, doc_end_map)

    # doc_name = '_'.join(os.path.basename(path).split('_')[:2])

    corpus_level_path = os.path.join(dir_path, '{}_corpus_level.conll'.format(doc_name))
    topic_level_path = os.path.join(dir_path, '{}_topic_level.conll'.format(doc_name))

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


    if corpus_level:
        doc_name = '_'.join(doc_name.split('_')[:2])
        with open(corpus_level_path, 'w') as f:
            f.write('#begin document {}\n'.format(doc_name))
            for token in corpus_level_tokens:
                f.write('\t'.join([str(x) for x in token]) + '\n')
            f.write('#end document')


    if topic_level:
        topic_level = collections.defaultdict(list)
        for token in corpus_level_tokens:
            topic = token[0]
            topic_level[topic].append(token)

        with open(topic_level_path, 'w') as f:
            for topic, tokens in topic_level.items():
                f.write('#begin document {}\n'.format(topic))
                for token in tokens:
                    f.write('\t'.join([str(x) for x in token]) + '\n')
                f.write('#end document\n')