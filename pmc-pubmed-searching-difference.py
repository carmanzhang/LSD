import math

import joblib
import pandas as pd
from Bio import Entrez
from aquarel import load_theme
from matplotlib import pyplot as plt
from scipy.stats import stats
from tqdm import tqdm

from metric.rank_rbo import rbo

based_dir = './'
all_themes = ['scientific', 'boxy_dark', 'boxy_light', 'umbra_light']
theme = load_theme(all_themes[3])
theme.apply()

Entrez.email = ""

df = pd.read_csv('pubmed_query_log.csv')

real_world_queries = df['query'].values

print('real_world_queries: ', real_world_queries)

# query_results = []
# for query in tqdm(real_world_queries):
#     # Note search PMC
#     print(query)
#     try:
#         handle = Entrez.esearch(db="pmc", retmax=2000, term=query, sort='relevance')
#         record = Entrez.read(handle)
#         handle.close()
#         pmc_ids = record["IdList"]
#
#         # Note search PubMed
#         handle = Entrez.esearch(db="pubmed", retmax=2000, term=query, sort='relevance')
#         record = Entrez.read(handle)
#         handle.close()
#         pm_ids = record["IdList"]
#         # print(pm_ids)
#         query_results.append([query, pmc_ids, pm_ids])
#
#         time.sleep(3)
#     except Exception as e:
#         print(e)
#
# joblib.dump(query_results, 'pmc-pubmed-article-query-result-comparison.pkl')
query_results = joblib.load('pmc-pubmed-article-query-result-comparison.pkl')

pmid_pmcid_mappings = pd.read_csv('pmid_pmcid_mappings.csv')


def rbo2(list1, list2, p=0.9):
    # tail recursive helper function
    def helper(ret, i, d):
        l1 = set(list1[:i]) if i < len(list1) else set(list1)
        l2 = set(list2[:i]) if i < len(list2) else set(list2)
        a_d = len(l1.intersection(l2)) / i
        term = math.pow(p, i) * a_d
        if d == i:
            return ret + term
        return helper(ret + term, i + 1, d)

    k = max(len(list1), len(list2))
    x_k = len(set(list1).intersection(set(list2)))
    summation = helper(0, 1, k)
    return ((float(x_k) / k) * math.pow(p, k)) + ((1 - p) / p * summation)


def rbo3(l1, l2, p=0.9):
    """
        Calculates Ranked Biased Overlap (RBO) score.
        l1 -- Ranked List 1
        l2 -- Ranked List 2
    """
    if l1 == None: l1 = []
    if l2 == None: l2 = []

    sl, ll = sorted([(len(l1), l1), (len(l2), l2)])
    s, S = sl
    l, L = ll
    if s == 0: return 0

    # Calculate the overlaps at ranks 1 through l
    # (the longer of the two lists)
    ss = set([])  # contains elements from the smaller list till depth i
    ls = set([])  # contains elements from the longer list till depth i
    x_d = {0: 0}
    sum1 = 0.0
    for i in range(l):
        x = L[i]
        y = S[i] if i < s else None
        d = i + 1

        # if two elements are same then
        # we don't need to add to either of the set
        if x == y:
            x_d[d] = x_d[d - 1] + 1.0
        # else add items to respective list
        # and calculate overlap
        else:
            ls.add(x)
            if y != None: ss.add(y)
            x_d[d] = x_d[d - 1] + (1.0 if x in ss else 0.0) + (1.0 if y in ls else 0.0)
            # calculate average overlap
        sum1 += x_d[d] / d * pow(p, d)

    sum2 = 0.0
    for i in range(l - s):
        d = s + i + 1
        sum2 += x_d[d] * (d - s) / (d * s) * pow(p, d)

    sum3 = ((x_d[l] - x_d[s]) / l + x_d[s] / s) * pow(p, l)

    # Equation 32
    rbo_ext = (1 - p) / p * (sum1 + sum2) + sum3
    return rbo_ext


metrics = []
top_n_list = list(range(1, 201, 1))
for top_n in tqdm(top_n_list):
    top_n_searching_metrics = []
    for query, pmc_ids, pm_ids in query_results:
        pmc_ids = ['PMC' + n for n in pmc_ids]
        mapped_pmc_ids = [pmid_pmcid_mappings[pmid] for pmid in pm_ids if pmid in pmid_pmcid_mappings]
        # print(query, len(pmc_ids), len(pm_ids), len(mapped_pmc_ids))
        if len(mapped_pmc_ids) < top_n:
            continue
        pmc_ids = pmc_ids[:top_n]
        mapped_pmc_ids = mapped_pmc_ids[:top_n]
        if len(pmc_ids) == 0 or len(mapped_pmc_ids) == 0:
            continue
        # https://towardsdatascience.com/rbo-v-s-kendall-tau-to-compare-ranked-lists-of-items-8776c5182899
        # https://raw.githubusercontent.com/dlukes/rbo/master/rbo.py
        # https://github.com/ragrawal/measures/blob/master/measures/rankedlist/RBO.py
        # rank-biased overlap
        rbo_v_method1 = rbo(pmc_ids, mapped_pmc_ids, p=0.9).min
        rbo_v_method2 = 0
        # rbo_v_method2 = rbo2(pmc_ids, mapped_pmc_ids, p=0.9)
        rbo_v_method3 = 0
        jaccard_v = len(set(pmc_ids).intersection(mapped_pmc_ids)) / len(set(pmc_ids).union(mapped_pmc_ids))
        min_len = min(len(pmc_ids), len(mapped_pmc_ids))
        # # note 比较两组排序的相关性 https://www.zhihu.com/tardis/zm/art/63279107?source_id=1003
        tau_v, p_v = stats.kendalltau(pmc_ids[:min_len], mapped_pmc_ids[:min_len])

        top_n_searching_metrics.append([rbo_v_method1, rbo_v_method2, rbo_v_method3, jaccard_v, tau_v])

    metrics.append(
        [sum([n[0] for n in top_n_searching_metrics]) / len(top_n_searching_metrics),
         sum([n[1] for n in top_n_searching_metrics]) / len(top_n_searching_metrics),
         sum([n[2] for n in top_n_searching_metrics]) / len(top_n_searching_metrics),
         sum([n[3] for n in top_n_searching_metrics]) / len(top_n_searching_metrics),
         sum([n[4] for n in top_n_searching_metrics]) / len(top_n_searching_metrics)]
    )

plt.plot(top_n_list, [n[0] for n in metrics], label='RBO', linewidth=3.5, marker='^', markevery=10)

plt.plot(top_n_list, [n[3] for n in metrics], label='Jaccard', linewidth=3.5, marker='o', markevery=10)
# plt.plot(top_n_list, [n[4] for n in metrics], label='kendalltau_v')
plt.xlim(1, top_n_list[-1] - 1)
plt.ylim(0, 0.2)

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(loc='best', prop={'size': 13})
plt.tight_layout()
plt.savefig(based_dir + '/figs/application-for-article-search.png', dpi=600)
plt.savefig(based_dir + '/figs/application-for-article-search.pdf')

plt.show()
