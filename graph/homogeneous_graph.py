import logging as log
import os
import time
from collections import Counter

import pandas as pd

from config import cached_dir


def flatten(d2_list):
    return [j for sub in d2_list for j in sub]


class PMCGraph:
    def __init__(self):
        pass

    def build_homogeneous_graph(self, df: pd.DataFrame, bidirectional=False, with_weight=False):
        # pm_id,author_id_arr,mesh_keywords,venue_nlm_unique_id,cited_pm_id_list
        citing_papers = list(df['pm_id'].values)
        log.info('number of involved citing papers: ' + str(len(citing_papers)))
        cited_papers = flatten(list(df['cited_pm_id_list'].values))

        log.info('number of involved cited papers: ' + str(len(cited_papers)))
        papers = sorted(list(set(citing_papers + cited_papers)))
        log.info('number of involved papers: ' + str(len(papers)))

        author_id_arr = df['author_id_arr'].values
        authors = sorted(list(set(flatten(author_id_arr))))  # flatten a 2D list to 1D, and deduplicate, then sorting
        log.info('number of involved authors: ' + str(len(authors)))

        mesh_keywords = df['mesh_keywords'].values
        keywords = sorted(list(set(flatten(mesh_keywords))))
        log.info('number of involved keywords: ' + str(len(keywords)))

        venue_nlm_unique_id = df['venue_nlm_unique_id'].values
        venues = sorted(list(set([n for n in venue_nlm_unique_id if len(n) > 0])))
        log.info('number of involved venues: ' + str(len(venues)))

        edge_list = []

        for _, row in df.iterrows():
            if len(row) != 5:
                log.info(('broken record', row))
            paper, paper_authors, kws, venue_id, cited_papers = row

            # P-P paper-(citing)>paper; paper-(cited)>paper
            for cited_paper in cited_papers:
                edge_list.append((paper, cited_paper) if paper <= cited_paper else (cited_paper, paper))

            # P-A paper-(authored)>author; author-(write)>paper;
            for author in paper_authors:
                edge_list.append((paper, author))

            # P-K paper-(has)>keywords; keywords-(in)>paper;
            for kw in kws:
                edge_list.append((paper, kw))

            # P-V
            edge_list.append((paper, venue_id))

            # # A-A
            # num_authors = len(paper_authors)
            # if num_authors > 1 and num_authors < 6:
            #     for i in range(num_authors - 1):
            #         for j in range(i + 1, num_authors):
            #             ai, aj = paper_authors[i], paper_authors[j]
            #             edge_list.append((ai, aj) if ai <= aj else (aj, ai))
            # # K-K
            # num_kws = len(kws)
            # if num_kws > 1 and num_kws <= 10:
            #     for i in range(num_kws - 1):
            #         for j in range(i + 1, num_kws):
            #             ki, kj = kws[i], kws[j]
            #             edge_list.append((ki, kj) if ki <= kj else (kj, ki))

        del df
        # compute edge weight
        print('num edge_list: ', len(edge_list))
        if bidirectional:
            edge_list = edge_list + [(b, a) for a, b in edge_list]

        spliter = ' '
        edge_list = [str(a) + spliter + str(b) for a, b in edge_list]
        edge_list = Counter(edge_list)
        edge_list = [edge + spliter + (str(freq) if with_weight else '') for edge, freq in edge_list.items()]
        print('num unique edge_list: ', len(edge_list))
        # hg = dgl.graph(edge_list)
        return edge_list


data_time_str = time.strftime('%Y-%m-%d-%H-%M')

if __name__ == '__main__':
    df = pd.read_csv('pmc_graph_element.csv')
    ###################################################################
    ## Note build the enhanced graph
    ###################################################################
    # "conference - paper - Author - paper - conference" metapath sampling
    pmc_graph = PMCGraph()
    edge_list = pmc_graph.build_homogeneous_graph(df, bidirectional=False, with_weight=False)
    print('have built the enhanced graph')

    fw = open(os.path.join(cached_dir, 'graph-embedding', 'enhanced-pmc-hetero-as-homo-graph-%s.tsv' % data_time_str), 'w')
    for edge_freq_str in edge_list:
        fw.write(edge_freq_str)
        fw.write("\n")
    fw.close()

    ###################################################################
    ## Note build the original graph
    ###################################################################
    pmc_graph = PMCGraph()
    # Note Do not use author ID and MeSH keywords!
    df['author_id_arr'] = df['author_id_arr'].map(lambda x: [])
    df['mesh_keywords'] = df['mesh_keywords'].map(lambda x: [])
    edge_list = pmc_graph.build_homogeneous_graph(df, bidirectional=False)
    print('have built the non-enhanced (original) graph')

    fw = open(os.path.join(cached_dir, 'graph-embedding', 'original-pmc-hetero-as-homo-graph-%s.tsv' % data_time_str), 'w')
    for edge_freq_str in edge_list:
        fw.write(edge_freq_str)
        fw.write("\n")
    fw.close()
