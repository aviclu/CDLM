import sys
from coval.conll import reader
from coval.conll import util
from coval.eval import evaluator


def main():
    allmetrics = [('mentions', evaluator.mentions), ('muc', evaluator.muc),
            ('bcub', evaluator.b_cubed), ('ceafe', evaluator.ceafe),
            ('lea', evaluator.lea)]

    key_file = sys.argv[1]
    sys_file = sys.argv[2]

    NP_only = 'NP_only' in sys.argv
    remove_nested = 'remove_nested' in sys.argv
    keep_singletons = ('remove_singletons' not in sys.argv
            and 'removIe_singleton' not in sys.argv)
    min_span = False
    if ('min_span' in sys.argv 
        or 'min_spans' in sys.argv
        or 'min' in sys.argv):
        min_span = True
        has_gold_parse = util.check_gold_parse_annotation(key_file)
        if not has_gold_parse:
                util.parse_key_file(key_file)
                key_file = key_file + ".parsed"


    if 'all' in sys.argv:
        metrics = allmetrics
    else:
        metrics = [(name, metric) for name, metric in allmetrics
                if name in sys.argv]
        if not metrics:
            metrics = allmetrics

    evaluate(key_file, sys_file, metrics, NP_only, remove_nested,
            keep_singletons, min_span)


def evaluate(key_file, sys_file, metrics, NP_only, remove_nested,
        keep_singletons, min_span):
    doc_coref_infos = reader.get_coref_infos(key_file, sys_file, NP_only,
            remove_nested, keep_singletons, min_span)

    conll = 0
    conll_subparts_num = 0

    for name, metric in metrics:
        recall, precision, f1 = evaluator.evaluate_documents(doc_coref_infos,
                metric,
                beta=1)
        if name in ["muc", "bcub", "ceafe"]:
            conll += f1
            conll_subparts_num += 1

        print(name.ljust(10), 'Recall: %.2f' % (recall * 100),
                ' Precision: %.2f' % (precision * 100),
                ' F1: %.2f' % (f1 * 100))

    if conll_subparts_num == 3:
        conll = (conll / 3) * 100
        print('CoNLL score: %.2f' % conll)


if __name__ == '__main__':
    main()
