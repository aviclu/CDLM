from os import walk
from os.path import isfile, join
from coval.arrau import markable

__author__ = 'ns-moosavi'


def get_doc_markables(doc_name, doc_lines, extract_MIN, word_column=0,
        markable_column=1, MIN_column=2, print_debug=False):
    markables_cluster = {}
    markables_start = {}
    markables_end = {}
    markables_MIN = {}
    markables_coref_tag = {}

    all_words = []

    for word_index, line in enumerate(doc_lines):
        columns = line.split()
        all_words.append(columns[word_column])

        # If the line contains annotations
        if len(columns) > 1:

            markable_annotations = columns[markable_column].split("@")
            MIN_annotations = columns[MIN_column].split(
                    "@") if extract_MIN and len(columns) >= 4 else None
            coref_annotations = columns[-1].split(
                    "@") if len(columns) >= 3 else None

            if print_debug:
                if ((MIN_annotations and len(markable_annotations)
                            != len(MIN_annotations))
                        or (coref_annotations and len(markable_annotations)
                            != len(coref_annotations))):
                    print((
                            'There is a problem with the annotation of the '
                            'document %r in line %s\n'
                            'The number of MIN or coref annotations '
                            'for each line should be equal to the the number '
                            'of markable annotations') % (doc_name, line))

            for i, markable_annotation in enumerate(markable_annotations):
                markable_id = int(markable_annotation[
                        11:markable_annotation.find('=')])
                cluster_id = int(markable_annotation[
                        markable_annotation.find('=') + 5:])

                if markable_annotation.startswith("B-markable_"):
                    markables_cluster[markable_id] = cluster_id
                    markables_start[markable_id] = word_index
                    markables_end[markable_id] = word_index

                    if MIN_annotations and len(markable_annotations) == len(
                            MIN_annotations) and MIN_annotations[i].strip():
                        if MIN_annotations[i].find('..') == -1:
                            MIN_start = int(MIN_annotations[i][5:]) - 1
                            MIN_end = MIN_start
                        else:
                            # -1 because word_index starts from zero
                            MIN_start = int(MIN_annotations[i][
                                    5:MIN_annotations[i].find('..')]) - 1
                            MIN_end = int(MIN_annotations[i][
                                    MIN_annotations[i].find('..') + 7:]) - 1
                        markables_MIN[markable_id] = (MIN_start, MIN_end)
                    else:
                        markables_MIN[markable_id] = None

                    if coref_annotations and len(markable_annotations) == len(
                            coref_annotations) and coref_annotations[i].strip(
                            ) == 'non_referring':
                        markables_coref_tag[markable_id] = 'non_referring'
                    else:
                        markables_coref_tag[markable_id] = 'referring'

                elif markable_annotation.startswith("I-markable_"):
                    markables_end[markable_id] = word_index

                else:
                    print((
                            '%r is not a valid annotation for markables.\n',
                            'The annotation of the following markable will be '
                            'skipped then.\n%s') % (markable_annotation, line))

    clusters = {}

    for markable_id in markables_cluster:
        m = markable.Markable(
                doc_name, markables_start[markable_id],
                markables_end[markable_id], markables_MIN[markable_id],
                markables_coref_tag[markable_id],
                all_words[markables_start[markable_id]:
                        markables_end[markable_id] + 1])

        if markables_cluster[markable_id] not in clusters:
            clusters[markables_cluster[markable_id]] = (
                    [], markables_coref_tag[markable_id])
        clusters[markables_cluster[markable_id]][0].append(m)

    return clusters


def process_clusters(clusters, keep_singletons, keep_non_referring):
    removed_non_referring = 0
    removed_singletons = 0
    processed_clusters = []
    processed_non_referrings = []

    for cluster_id, (cluster, ref_tag) in clusters.items():
        if ref_tag == 'non_referring':
            if keep_non_referring:
                processed_non_referrings.append(clusters[cluster_id][0][0])
            else:
                removed_non_referring += 1
            continue
        if not keep_singletons and len(cluster) == 1:
            removed_singletons += 1
            continue

        processed_clusters.append(clusters[cluster_id][0])

    return (processed_clusters, processed_non_referrings,
            removed_non_referring, removed_singletons)


def get_coref_infos(key_directory,
        sys_directory,
        keep_singletons,
        keep_non_referring,
        use_MIN,
        print_debug=False):

    key_docs = get_all_docs(key_directory)
    sys_docs = get_all_docs(sys_directory)

    doc_coref_infos = {}
    doc_non_referrig_infos = {}

    for doc in key_docs:

        if doc not in sys_docs:
            print('The document ', doc,
                    ' does not exist in the system output.')
            continue

        key_clusters = get_doc_markables(doc, key_docs[doc], use_MIN)
        sys_clusters = get_doc_markables(doc, sys_docs[doc], False)

        (key_clusters, key_non_referrings, key_removed_non_referring,
                key_removed_singletons) = process_clusters(
                key_clusters, keep_singletons, keep_non_referring)
        (sys_clusters, sys_non_referrings, sys_removed_non_referring,
                sys_removed_singletons) = process_clusters(
                sys_clusters, keep_singletons, keep_non_referring)

        sys_mention_key_cluster = get_markable_assignments(
                sys_clusters, key_clusters)
        key_mention_sys_cluster = get_markable_assignments(
                key_clusters, sys_clusters)

        doc_coref_infos[doc] = (key_clusters, sys_clusters,
                    key_mention_sys_cluster, sys_mention_key_cluster)
        doc_non_referrig_infos[doc] = (key_non_referrings, sys_non_referrings)

        if print_debug and not keep_non_referring:
            print('%s and %s non-referring markables are removed from the '
                    'evaluations of the key and system files, respectively.'
                    % (key_removed_non_referring, sys_removed_non_referring))

        if print_debug and not keep_singletons:
            print('%s and %s singletons are removed from the evaluations of '
                    'the key and system files, respectively.'
                    % (key_removed_singletons, sys_removed_singletons))

    return doc_coref_infos, doc_non_referrig_infos


def get_markable_assignments(inp_clusters, out_clusters):
    markable_cluster_ids = {}
    out_dic = {}
    for cluster_id, cluster in enumerate(out_clusters):
        for m in cluster:
            out_dic[m] = cluster_id

    for cluster in inp_clusters:
        for im in cluster:
            for om in out_dic:
                if im == om:
                    markable_cluster_ids[im] = out_dic[om]
                    break

    return markable_cluster_ids


def get_all_docs(path):
    all_docs = {}
    if isfile(path):
        if path.endswith('.CONLL'):
            all_docs[path[path.rfind('/') + 1:]] = get_doc_lines(path)
    else:
        for root, _directories, filenames in walk(path):
            for filename in filenames:
                if (filename.endswith('.CONLL')):
                    all_docs[filename] = get_doc_lines(join(root, filename))
    return all_docs


def get_doc_lines(file_name):
    doc_lines = []

    with open(file_name) as f:
        for line in f:
            if line.startswith("TOKEN"):
                continue
            doc_lines.append(line)

    return doc_lines