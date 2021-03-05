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
    sys_file = 'data/ecb/gold/test_{}_corpus_level.conll'.format(mention_type)

    print('Processing file: {0}'.format(path))
    full_path = path
    scores = evaluate(full_path, sys_file, allmetrics, NP_only, remove_nested,
            keep_singletons, min_span)


    print(scores)


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
