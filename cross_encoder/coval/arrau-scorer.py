import sys
from coval.arrau import reader
from coval.eval import evaluator
from coval.eval.evaluator import evaluate_non_referrings

__author__ = 'ns-moosavi'


def main():
    metric_dict = {
            'lea': evaluator.lea, 'muc': evaluator.muc,
            'bcub': evaluator.b_cubed, 'ceafe': evaluator.ceafe}
    key_directory = sys.argv[1]
    sys_directory = sys.argv[2]

    if 'remove_singletons' in sys.argv or 'remove_singleton' in sys.argv:
        keep_singletons = False
    else:
        keep_singletons = True

    if 'MIN' in sys.argv or 'min' in sys.argv or 'min_spans' in sys.argv:
        use_MIN = True
    else:
        use_MIN = False

    if 'keep_non_referring' in sys.argv or 'keep_non_referrings' in sys.argv:
        keep_non_referring = True
    else:
        keep_non_referring = False

    if 'all' in sys.argv:
        metrics = [(k, metric_dict[k]) for k in metric_dict]
    else:
        metrics = []
        for name in metric_dict:
            if name in sys.argv:
                metrics.append((name, metric_dict[name]))

    if len(metrics) == 0:
        metrics = [(name, metric_dict[name]) for name in metric_dict]

    msg = ""
    if keep_non_referring and keep_singletons:
        msg = ('all annotated markables, i.e. including corferent markables, '
                'singletons and non-referring markables')
    elif keep_non_referring and not keep_singletons:
        msg = ('only coreferent markables and non-referring markables, '
                'excluding singletons')
    elif not keep_non_referring and keep_singletons:
        msg = ('coreferring markables and singletons, '
                'excluding non-referring mentions.')
    else:
        msg = ('only coreferring markables, '
                'excluding singletons and non-referring mentions')

    print('The scorer is evaluating ', msg,
            ("using the minimum span evaluation setting " if use_MIN else ""))

    evaluate(key_directory, sys_directory, metrics, keep_singletons,
            keep_non_referring, use_MIN)


def evaluate(key_directory, sys_directory, metrics, keep_singletons,
        keep_non_referring, use_MIN):

    doc_coref_infos, doc_non_referring_infos = reader.get_coref_infos(
            key_directory, sys_directory, keep_singletons, keep_non_referring,
            use_MIN)

    conll = 0
    conll_subparts_num = 0

    for name, metric in metrics:
        recall, precision, f1 = evaluator.evaluate_documents(doc_coref_infos,
                metric,
                beta=1)
        if name in ["muc", "bcub", "ceafe"]:
            conll += f1
            conll_subparts_num += 1

        print(name)
        print('Recall: %.2f' % (recall * 100),
                ' Precision: %.2f' % (precision * 100),
                ' F1: %.2f' % (f1 * 100))

    if conll_subparts_num == 3:
        conll = (conll / 3) * 100
        print('CoNLL score: %.2f' % conll)

    if keep_non_referring:
        recall, precision, f1 = evaluate_non_referrings(
                doc_non_referring_infos)
        print('============================================')
        print('Non-referring markable identification scores:')
        print('Recall: %.2f' % (recall * 100),
                ' Precision: %.2f' % (precision * 100),
                ' F1: %.2f' % (f1 * 100))


main()
