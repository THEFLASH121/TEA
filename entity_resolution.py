import argparse
import datetime as dt
import pickle
import args
import gensim.models as models
from tqdm import tqdm

from utils import *

NGT_NOT_FOUND = ANNOY_NOT_FOUND = FAISS_NOT_FOUND = False

try:
    import ngtpy
except ModuleNotFoundError:
    warnings.warn('ngtpy not found. NGT indexing will not be available.')
    NGT_NOT_FOUND = True

try:
    import faiss
except ModuleNotFoundError:
    warnings.warn('faiss not found. faiss indexing will not be available.')
    FAISS_NOT_FOUND = True

try:
    from gensim.similarities.index import AnnoyIndexer
except ImportError:
    warnings.warn('AnnoyIndexer not found. Annoy indexing will not be available.')
    ANNOY_NOT_FOUND = True



def parse_args():
    '''Argument parser for standalone execution. 
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', action='store', required=True, type=str)
    parser.add_argument('-m', '--matches_file', action='store', required=True, type=str)
    parser.add_argument('--n_top', default=5, type=int, help='Number of neighbors to choose from.')
    parser.add_argument('--n_candidates', default=1, type=int, help='Number of candidates to choose among the n_top '
                                                                    'neighbors.')
    parser.add_argument('--info_file', default='', type=str, required=True)

    return parser.parse_args()


# target是遍历A表的每一个idx，most_similar是匹配预测值
def _check_symmetry(target, most_similar, n_top):
    valid = []
    for cand in most_similar[target]:
        mm = most_similar[cand]
        if target in mm[:n_top]:
            valid.append(cand)
    return valid

#model_file是所有idx的embedding文件路径
#viable_lines是一个list，包含所有idx的embedding的值
def build_similarity_structure(model_file, viable_lines, n_items, strategy,
                               n_top=10, n_candidates=1, num_trees=None, epsilon=None):
    t_start = dt.datetime.now()
    most_similar = {}
    c = 1

    #所有的idx节点集合
    nodes = [_.split(' ', maxsplit=1)[0] for _ in viable_lines]
    # viable_lines = []

    if strategy == 'annoy' and ANNOY_NOT_FOUND:
        warnings.warn('Chosen strategy = \'annoy\', but the module is not installed. Falling back to basic.')
        strategy = 'basic'
    if strategy == 'ngt' and NGT_NOT_FOUND:
        warnings.warn('Chosen strategy = \'NGT\', but the module is not installed. Falling back to basic.')
        strategy = 'basic'
    if strategy == 'faiss' and FAISS_NOT_FOUND:
        warnings.warn('Chosen strategy = \'faiss\', but the module is not installed. Falling back to basic.')
        strategy = 'basic'
#-----------------------------------------------------------------------------------------------------------------
    if strategy == 'basic':
        model = models.KeyedVectors.load_word2vec_format(model_file, unicode_errors='ignore')

        for n in tqdm(nodes, desc='# ER - Finding node matches: '):
            ms = model.most_similar(str(n), topn=n_top)
            mm = [item[0] for item in ms]
            idx = int(n.split('__')[1])

            # n_items是表A的行数，这里就是做一个该元素属于A表还是B表的判断：
            if idx < n_items:# 属于A表
                candidates = [_ for _ in mm if int(_.split('__')[1]) >= n_items]
            else:# 属于B表
                candidates = [_ for _ in mm if int(_.split('__')[1]) < n_items]
            #上面if-else语句完成了什么事呢？如果当前节点属于A表，那么10个候选表中属于B表的作为candidates，else反之。

            candidates = candidates[:n_candidates]
            most_similar[n] = candidates
            c += 1
        print('')
# -----------------------------------------------------------------------------------------------------------------
    elif strategy == 'annoy':
        assert num_trees is not None
        assert type(num_trees) == int
        assert num_trees > 0

        print('Using ANNOY indexing.')
        model = models.KeyedVectors.load_word2vec_format(model_file, unicode_errors='ignore')
        annoy_index = AnnoyIndexer(model, num_trees=num_trees)
        for n in tqdm(nodes):
            ms = model.most_similar(str(n), topn=n_top, indexer=annoy_index)
            mm = [item[0] for item in ms]
            idx = int(n.split('__')[1])
            if idx < n_items:
                candidates = [_ for _ in mm if int(_.split('__')[1]) >= n_items]
            else:
                candidates = [_ for _ in mm if int(_.split('__')[1]) < n_items]

            candidates = candidates[:n_candidates]
            most_similar[n] = candidates
            print('\rBuilding similarity structure: {:0.1f} - {}/{} tuples'.format(c / len(nodes) * 100, c, len(nodes)),
                  end='')
            c += 1
        print('')
# -----------------------------------------------------------------------------------------------------------------
    elif strategy == 'ngt':
        assert epsilon is not None
        assert type(epsilon) == float
        assert 0 <= epsilon <= 1

        print('Using NGT indexing.')
        ngt_index_path = 'pipeline/dump/ngt_index.nn'
        words = []
        with open(model_file, 'r') as fp:
            n, dim = map(int, fp.readline().split())
            ngtpy.create(ngt_index_path, dim, distance_type='Cosine')
            index = ngtpy.Index(ngt_index_path)

            for idx, line in enumerate(fp):
                k, v = line.rstrip().split(' ', maxsplit=1)
                vector = list(map(float, v.split(' ')))
                index.insert(vector)  # insert objects
                words.append(k)

        index.build_index()
        index.save()
        most_similar = {}

        for n in tqdm(nodes):
            query = index.get_object(words.index(n))
            ms = index.search(query, size=n_top, epsilon=epsilon)
            mm = [item[0] for item in ms[1:]]
            mm = list(map(words.__getitem__, mm))
            idx = int(n.split('_')[1])
            if idx < n_items:
                candidates = [_ for _ in mm if idx >= n_items]
            else:
                candidates = [_ for _ in mm if idx < n_items]

            candidates = candidates[:n_candidates]
            most_similar[n] = candidates
            print('\rBuilding similarity structure: {:0.1f} - {}/{} tuples'.format(c / len(nodes) * 100, c, len(nodes)),
                  end='')
            c += 1
        print('')
# -----------------------------------------------------------------------------------------------------------------
    elif strategy == 'faiss':
        print('Using faiss indexing.')
        # ngt_index_path = 'pipeline/dump/ngt_index.nn'
        words = []
        with open(model_file, 'r') as fp:
            n, dim = map(int, fp.readline().split())
            mat = []
            index = faiss.IndexFlatL2(dim)
            for idx, line in enumerate(fp):
                k, v = line.rstrip().split(' ', maxsplit=1)
                vector = np.array(list(map(float, v.split(' '))), ndmin=1).astype('float32')
                mat.append(vector)
                words.append(k)

        mat = np.array(mat)
        index.add(mat)

        most_similar = {}

        D, I = index.search(mat, k=n_top+1)
        # D, I = index.search(query, size=n_top, epsilon=epsilon)
        # mm = [item[0] for item in ms[1:]]
        # mm = list(map(words.__getitem__, mm))
        for n in tqdm(nodes):
            idx = int(n.split('__')[1])
            mm = I[idx]
            if idx < n_items:
                candidates = [_ for _ in mm if idx >= n_items]
            else:
                candidates = [_ for _ in mm if idx < n_items]

            candidates = candidates[:n_candidates]
            most_similar[n] = ['idx__{}'.format(_) for _ in candidates]
        # print('\rBuilding similarity structure: {:0.1f} - {}/{} tuples'.format(c / len(nodes) * 100, c, len(nodes)),
        #       end='')
        c += 1
        print('')



    else:
        raise ValueError('Unknown strategy {0}'.format(strategy))

    t_end = dt.datetime.now()
    diff = t_end - t_start
    print('# Time required to build sim struct: {:.2} seconds'.format(diff.total_seconds()))
    pickle.dump(most_similar, open('most_similar.pickle', 'wb'))

    return most_similar


def compare_ground_truth_only(most_similar, matches_file, n_items, n_top):
    """
    Test the accuracy of matches by
    :param most_similar:这是匹配预测值
    :param matches_file:这是ground truth
    :param n_items:这是A表的行数（通常用来区分表A和表B的一个分界）
    :param n_top:
    """
    #把 match文件读成一个字典：
    matches = _read_matches(matches_file)

    in_ground_truth = set()
    for tup in matches.items():
        tmp = [tup[0]] + tup[1]
        for _ in tmp:
            in_ground_truth.add(_)


    prefix = list(matches.keys())[0].split('_')[0]

    count_miss = count_hit = 0
    iteration_counter = 0
    total_candidates = no_candidate_found = 0
    false_candidates = 0
    golden_candidates = 0

    csvfile = open('suspicious_matches.csv', 'w')
    csvwriter = csv.writer(csvfile, delimiter=',')
    csvwriter.writerow(['id1','id2'])

    for n in range(n_items):#遍历表A的范围
        item = prefix + '__' + str(n)#表A中的每一行
        # Take the n_top closest neighbors of item
        try:
            # Extract only the name of the neighbors
            candidates = _check_symmetry(item, most_similar, n_top)#对于表A中每一行，去看匹配预测里有没有对应的两条记录（eg：idx1->idx5;idx5->idx1，这就算一个candidate，含义：确保预测这两条不矛盾，A表的ER与B表的ER相对应，然后再次基础上再与ground truth比）
            #得跟ground truth比：
            if item in matches:# 遍历A表的每一行，如果该idx在ground truth里，才能进行下一步验证
                for val in candidates: #对于每一个candidate
                    if val in matches[item]:#如果该candidate在ground truth里，就认为是hit
                        count_hit += 1#hit是什么含义？预测正确！
                    else:
                        count_miss += 1
                if len(candidates) == 0: no_candidate_found += 1
                total_candidates += len(candidates)# 所有符合条件（自身不矛盾）的预测值
                false_candidates += len(candidates)
            else:
                for val in candidates:
                    if val in in_ground_truth:
                        count_miss += 1
                        total_candidates += 1
            golden_candidates+=len(candidates)
        except KeyError:
            if item in matches: count_miss += 1

        iteration_counter += 1
    if total_candidates < 1: precision = 0
    else: precision = count_hit / total_candidates# 准确率公式
    recall = count_hit / len(matches)# 召回率公式
    try:
        f1_score = 2 * (precision * recall) / (precision + recall)# F1-Score公式
    except ZeroDivisionError:
        f1_score = 0

    if golden_candidates < 1: golden_precision = 0
    else: golden_precision = count_hit / golden_candidates
    try:
        golden_f1 = 2 * (golden_precision * recall) / (golden_precision + recall)
    except ZeroDivisionError:
        golden_f1 = 0

    print('# Total candidates tested: {}'.format(total_candidates))
    print('# {} cases where no candidates were found'.format(no_candidate_found))
    result_dict = {
        'P' : precision,
        'R' : recall,
        'F' : f1_score,
        'GP': golden_precision,
        'GR': recall,
        'GF': golden_f1,
    }

    print('\nP\tR\tF\tGP\tGR\tGF')
    for _ in result_dict.values():
        print('{:.4f}\t'.format(_*100), end='')
    print('\r')
    # print(s.format([_*100 for _ in ]))
    print('\n# Correct: {}\n# Wrong: {}\n# Total items: {}\n# Total matches: {}'.format(count_hit, count_miss,
                                                                              iteration_counter, len(matches)))
    return result_dict


def perform_matching(most_similar):
    matches = []
    for idx in most_similar:
        for m in most_similar[idx]:
            i1 = idx.split('__')[1]
            i2 = m.split('__')[1]
            t = sorted([i1, i2])
            matches.append(tuple(['idx__{}'.format(_) for _ in t]))
    return matches


def entity_resolution(input_file: str,
                                 task: str = 'test', info_file: str = None):
    t_start = dt.datetime.now()

    # n_top = configuration['ntop']# n_closest_neighbors_to_study
    # n_candidates = configuration['ncand'] #在寻找最近邻节点时，从ntop中选择的候选节点数量。提高召回率，大大降低精确度
    # strategy = configuration['indexing']
    # matches_file = configuration['match_file']
    n_top = args.ntop# n_closest_neighbors_to_study
    n_candidates = args.ncand
    strategy = args.indexing
    matches_file = args.match_file



    model_file, viable_lines = _prepare_tests(input_file)# viale_lines是所有idx的embedding ； model_file是这个viable_lines文件的路径
    with open(info_file, 'r', encoding='utf-8') as fp:
        line = fp.readline()
        n_items = int(line.split(',')[1]) # 只读了表A的行数

    # most_similar就是为每个idx找到与他最匹配的那个idx（A表和B表之间的匹配）
    most_similar = build_similarity_structure(model_file, viable_lines, n_items, strategy, n_top, n_candidates,
                                              epsilon=args.epsilon, num_trees=args.num_trees)
    if task == 'test':
        dict_result = compare_ground_truth_only(most_similar, matches_file, n_items, n_top)#对上面匹配结果进行验证
    elif task == 'match':
        matches = perform_matching(most_similar)
    else:
        raise ValueError('Unknown task {}'.format(task))

    t1 = dt.datetime.now()
    str_start_time = t1.strftime(TIME_FORMAT)
    t_end = dt.datetime.now()
    diff = t_end - t_start
    print('# Time required to execute the ER task: {}'.format(diff.total_seconds()))

    if task == 'test':
        return dict_result
    elif task == 'match':
        return matches
    else:
        return None


def _read_matches(matches_file):
    matches = {}
    n_lines = 0
    with open(matches_file, 'r', encoding='utf-8') as fp:
        for n, line in enumerate(fp.readlines()):
            if len(line.strip()) > 0:
                item, match = line.replace('_', '__').split(',')
                if item not in matches:
                    matches[item] = [match.strip()]
                else:
                    matches[item].append(match.strip())
                n_lines = n
        if n_lines == 0:
            raise IOError('Matches file is empty. ')
    return matches


def _prepare_tests(model_file):
    with open(model_file, 'r', encoding='utf-8') as fp:
        s = fp.readline()
        _, dimensions = s.strip().split(' ')
        viable_idx = []
        for i, row in enumerate(fp):
            if i >= 0:
                idx, vec = row.split(' ', maxsplit=1)
                if idx.startswith('idx__'):
                    try:
                        prefix, n = idx.split('__')
                        n = int(n)
                    except ValueError:
                        continue
                    viable_idx.append(row)
        # viable_idx = [row for idx, row in enumerate(fp) if idx > 0 and row.startswith('idx_')]

    f = 'dump/indices.emb'
    with open(f, 'w', encoding='utf-8') as fp:
        fp.write('{} {}\n'.format(len(viable_idx), dimensions))
        for _ in viable_idx:
            fp.write(_)

    return f, viable_idx
