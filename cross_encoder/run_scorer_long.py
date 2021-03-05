import sys
from coval.coval.conll import reader
from coval.coval.conll import util
from coval.coval.eval import evaluator
import pandas as pd
import os
from utils import *


def main():
    allmetrics = [('mentions', evaluator.mentions), ('muc', evaluator.muc),
            ('bcub', evaluator.b_cubed), ('ceafe', evaluator.ceafe),
            ('lea', evaluator.lea)]

    NP_only = 'NP_only' in sys.argv
    remove_nested = 'remove_nested' in sys.argv
    keep_singletons = ('remove_singletons' not in sys.argv
                       and 'removIe_singleton' not in sys.argv)
    min_span = False

    path = sys.argv[1]
    mention_type = sys.argv[2]
    sys_file = 'data/ecb/gold/dev_{}_topic_level.conll'.format(mention_type)

    all_scores = {}
    max_conll_f1 = (None, 0)
    for subdir in os.listdir(path):
        if 'checkpoint' not in subdir:
            continue
        subdir_path = os.path.join(path, subdir)
        for key_file in os.listdir(subdir_path):
            if key_file.endswith('conll') and key_file.startswith('dev'):
                print('Processing file: {0} for {1}'.format(key_file, subdir))
                full_path = os.path.join(subdir_path,key_file)
                scores = evaluate(full_path, sys_file, allmetrics, NP_only, remove_nested,
                        keep_singletons, min_span)
                key_file_name = key_file+'_'+subdir
                all_scores[key_file_name] = scores
                if scores['conll'] > max_conll_f1[1]:
                    max_conll_f1 = (key_file_name, scores['conll'])

    df = pd.DataFrame.from_dict(all_scores)
    df.to_csv(os.path.join(path, 'all_scores.csv'))


    print(max_conll_f1)


def evaluate(key_file, sys_file, metrics, NP_only, remove_nested,keep_singletons, min_span):
    doc_coref_infos = reader.get_coref_infos(key_file, sys_file, NP_only,remove_nested, keep_singletons, min_span)

    conll = 0
    conll_subparts_num = 0

    scores = {}

    for name, metric in metrics:
        recall, precision, f1 = evaluator.evaluate_documents(doc_coref_infos,
                metric,
                beta=1)

        scores['{}_{}'.format(name, 'recall')] = recall
        scores['{}_{}'.format(name, 'precision')] = precision
        scores['{}_{}'.format(name, 'f1')] = f1

        if name in ["muc", "bcub", "ceafe"]:
            conll += f1
            conll_subparts_num += 1

    scores['conll'] = (conll / 3) * 100


    return scores

if __name__ == '__main__':

    main()
