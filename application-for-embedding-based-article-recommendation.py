import json
import pickle
import sys
import time
from collections import Counter

import faiss
import numpy as np
import pandas as pd

from metric import rank_metrics

based_path = './'

task_id = 0
tasks = {0: 'original-pmc-hetero-as-homo-graph-2023-12-24-15-05.tsv',
         1: 'enhanced-pmc-hetero-as-homo-graph-2023-12-24-15-05.tsv'}
task_name = tasks[task_id]

D = 64
topk = 2000

field_split = '#'
top_n_interval = 20
set_list_length = 100

# N = 10000
num_batch_lines = 7000000

paper_element_id_map = pickle.load(open(based_path + task_name + '.idmap', 'rb'))
paper_id_element_map = {v: k for k, v in paper_element_id_map.items()}
embedding_data = np.load(based_path + task_name + '.emb.npy')
# embedding_data = embedding_data[[v for k,v in idmap.items() if k.startswith('p')]]

df_test = pd.read_csv('pmc_graph_element_test.csv')

df_test['id'] = df_test['pm_id'].apply(lambda x: paper_element_id_map[x])
df_test['reference_ids'] = df_test['reference'].apply(lambda x: [paper_element_id_map[n] for n in x])


class ResultHeap:
    """ Combine query results from a sliced dataset """

    def __init__(self, nq, k):
        " nq: number of query vectors, k: number of results per query "
        self.I = np.zeros((nq, k), dtype='int64')
        self.D = np.zeros((nq, k), dtype='float32')
        self.nq, self.k = nq, k
        heaps = faiss.float_maxheap_array_t()
        heaps.k = k
        heaps.nh = nq
        heaps.val = faiss.swig_ptr(self.D)
        heaps.ids = faiss.swig_ptr(self.I)
        heaps.heapify()
        self.heaps = heaps

    def add_batch_result(self, D, I, i0):
        assert D.shape == (self.nq, self.k)
        assert I.shape == (self.nq, self.k)
        I += i0
        self.heaps.addn_with_ids(
            self.k, faiss.swig_ptr(D),
            faiss.swig_ptr(I), self.k)

    def finalize(self):
        self.heaps.reorder()


def parse_vectors_with_ids(path, sx, ex):
    with open(path, 'r') as f:
        for i in range(sx):
            next(f)
        lines = [f.readline().strip().split('\t') for _ in range(ex - sx)]
        # print((len(lines), lines[0]))
        embs = np.array([[float(m) for m in n[1:]] for n in lines], np.float32)
        ids = np.array([n[0] for n in lines])
    return ids, embs


# compute by blocks of bs, and add to heaps
def search_sliced(base_embs_slice, query_embs, topk, i0, num_instances):
    t0 = time.time()
    # xsl = sanitize(base_embs_slice)
    index.add(base_embs_slice)
    D, I = index.search(query_embs, topk)
    rh.add_batch_result(D, I, i0)
    index.reset()
    print("\r   %d/%d, %.3f s" % (i0, num_instances, time.time() - t0), end=' ')
    sys.stdout.flush()
    print()


def sanitize(x):
    return np.ascontiguousarray(x, dtype='float32')


# step 0, read the number of lines of query and repository
print('read numbers of lines of repository')
num_queries = len(df_test)
num_query_batch = int(np.ceil(num_queries / num_batch_lines))
num_lines = len(embedding_data)
print('num queries: ', num_queries, 'num items: ', num_lines, 'num_query_batch: ', num_query_batch)

# step 1, create gpu based index instance
print('create gpu based index instance')
# index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(D))
# IP stands for "inner product" == cosine similarity.
index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatIP(D))  # 归一化的向量点乘即cosine相似度（越大越好）

# step 2. read all query embeddings
print('read all query embeddings')
i_starts = list(range(0, num_lines, num_batch_lines))
i_ends = i_starts[1:] + [num_lines]
print(i_starts, i_ends)

q_starts = list(range(0, num_queries, num_batch_lines))
q_ends = q_starts[1:] + [num_queries]
print(q_starts, q_ends)


def pretty_metrics(kv: list, decimal=2, pctg=False, sep=field_split):
    k = [item[0] for item in kv]
    if pctg:
        v = [round(item[1] * 100.0, decimal) for item in kv]
    else:
        v = [round(item[1], decimal) for item in kv]
    if not sep:
        df = pd.DataFrame(data=[v], columns=k)
        return df.head()
    else:
        return sep.join([str(s) for s in v])


def get_ranking_metric_values(all_query_ranks, check_length=False, list_length=set_list_length, check_sum=False,
                              list_sum=1):
    l = len(all_query_ranks)
    if check_sum:
        all_query_ranks = [n for n in all_query_ranks if sum(n) == list_sum]
        if l != len(all_query_ranks):
            print('remove sum value broken query result: ', l - len(all_query_ranks))
    l = len(all_query_ranks)
    if check_length:
        all_query_ranks = [n for n in all_query_ranks if len(n) == list_length]
        if l != len(all_query_ranks):
            print('remove length value broken query result: ', l - len(all_query_ranks))

    print('query distribution: ', Counter([len(n) for n in all_query_ranks]))
    topns, num_queries, maps, mrrs, ndcgs, hitrates = [], [], [], [], [], []
    for i, top_n in enumerate(list(range(top_n_interval, set_list_length + 1, top_n_interval))):
        # query result 的长度不一致，会对结果不公平，这里过滤掉这样的结果，截断过长的结果，只取topn个结果
        all_query_ranks_topn = [n[:top_n] for n in all_query_ranks]
        num_query = len(all_query_ranks_topn)
        # print('evaluate %s queries after delete shorter query' % len(all_query_ranks))
        map = rank_metrics.mean_average_precision(all_query_ranks_topn)
        mrr = rank_metrics.mean_reciprocal_rank(all_query_ranks_topn)
        ndcg = np.average([rank_metrics.ndcg_at_k(r, k=top_n) for r in all_query_ranks_topn])

        hitrate = np.nan if len(all_query_ranks_topn) == 0 else np.count_nonzero(
            np.sum(np.array(all_query_ranks_topn), axis=1)) / len(all_query_ranks_topn)

        topns.append(top_n)
        num_queries.append(num_query)
        maps.append(map)
        mrrs.append(mrr)
        ndcgs.append(ndcg)
        hitrates.append(hitrate)

    # MRR only cares about the single highest-ranked relevant item.
    # When there is only one relevant answer in your dataset,
    # the MRR and the MAP are exactly equivalent under the standard definition of MAP.
    ranking_metrics_in_string = json.dumps({
        'top_n': pretty_metrics(list(zip(topns, topns)), decimal=4, pctg=False, sep=field_split),
        'num_query': pretty_metrics(list(zip(topns, num_queries)), decimal=4, pctg=False, sep=field_split),
        'map': pretty_metrics(list(zip(topns, maps)), decimal=4, pctg=False, sep=field_split),
        'hitrates': pretty_metrics(list(zip(topns, hitrates)), decimal=4, pctg=False, sep=field_split),
        # 'mrr': pretty_metrics(list(zip(topns, mrrs)), decimal=4, pctg=False, sep=field_split),
        'ndcg': pretty_metrics(list(zip(topns, ndcgs)), decimal=4, pctg=False, sep=field_split)
    }, indent=4, sort_keys=True).replace('#', '\t').replace('": "', '\t').replace('"', '')

    return topns, num_queries, maps, hitrates, ndcgs, ranking_metrics_in_string


for idx, (sx, ex) in enumerate(zip(q_starts, q_ends)):
    print("execute num of batches %d/%d" % (idx, num_query_batch))
    query_ids = df_test['id'].values
    ground_truths = df_test['reference_ids'].values
    query_embs = np.array([embedding_data[n] for n in query_ids])
    print('query_ids query_embs shape: ', query_ids.shape, query_embs.shape)

    # step 3. create ResultHeap that store the result of each slice, combine them into a global search result
    print('create ResultHeap')
    num_query_instance = len(query_ids)
    rh = ResultHeap(num_query_instance, topk)

    # step 4. compute by batches, and add to result heaps
    # print('compute by batches, and add to result heaps')
    bc = 0
    base_ids = []

    for idxx, (i0, i1) in enumerate(zip(i_starts, i_ends)):
        bc += 1
        print("execute %d/%d" % (i0, num_lines))
        # i1 = min(num_batches, i0 + num_batch_lines)
        base_ids_slice = np.array(list(range(i0, i1)))
        base_embs_slice = np.array(embedding_data[i0:i1], np.float32)
        print('shape: ', base_ids_slice.shape, base_embs_slice.shape)
        # base_ids_slice = base_ids_slice
        base_ids.extend(base_ids_slice)
        base_embs_slice = sanitize(base_embs_slice)
        search_sliced(base_embs_slice=base_embs_slice, query_embs=query_embs, topk=topk, i0=i0,
                      num_instances=num_lines)

    # step 5. finalize ResultHeap
    print('finalize ResultHeap')
    rh.finalize()
    D, I = rh.D, rh.I

    print(D.shape, I.shape)
    # print(D[:10])
    # print(D[-10:])
    #
    # print(I[:10])
    # print(I[-10:])

    assert len(query_ids) == I.shape[0]
    assert I.shape[1] == topk

    query_results = []
    for ii, query_id in enumerate(query_ids):
        ground_truth = ground_truths[ii]
        topk_query_result = [base_ids[j] for j in I[ii][::-1]]  # in descending order
        topk_query_result = [r for r in topk_query_result if paper_id_element_map[r].startswith('p')]  # 只保留关于论文的检索结果

        if query_id in topk_query_result:
            topk_query_result.remove(query_id)
        else:
            print('query id not in query results')
        # print(len(set(topk_query_result).intersection(ground_truth)))
        query_results.append([query_id, ground_truth, topk_query_result])

    pd.DataFrame(query_results, columns=['query_id', 'ground_truth', 'topk_query_result']).to_csv(
        based_path + tasks[task_id] + '.queryresult', index=False)

    all_query_ranks = []
    for query_id, ground_truth, topk_query_result in query_results:
        ground_truth = set(ground_truth)
        hits = [1 if n in ground_truth else 0 for n in topk_query_result]
        all_query_ranks.append(hits)

    topns, num_queries, maps, mrrs, ndcgs, ranking_metrics_in_string = get_ranking_metric_values(all_query_ranks)
    print('*' * 100)
    print(task_name)
    print(topns, num_queries, maps, mrrs, ndcgs)
    print(ranking_metrics_in_string)
