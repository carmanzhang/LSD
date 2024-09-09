import os

from config import cached_dir

data_time_str = '2023-12-24-15-05'
fr = open(os.path.join(cached_dir, 'graph-embedding', 'enhanced-pmc-hetero-as-homo-graph-%s.tsv' % data_time_str), 'r')

edge_count = 0
vertex_set = set()

for edge_freq_str in fr:
    edge_count += 1
    v1, v2 = edge_freq_str.strip().split(' ')
    vertex_set.add(v1)
    vertex_set.add(v2)

print('enhanced-graph: number of edges: %d, number of vertex: %d, density: %.2f' % (
edge_count, len(vertex_set), edge_count / len(vertex_set)))

edge_count = 0
vertex_set = set()

fr = open(os.path.join(cached_dir, 'graph-embedding', 'original-pmc-hetero-as-homo-graph-%s.tsv' % data_time_str), 'r')
for edge_freq_str in fr:
    edge_count += 1
    v1, v2 = edge_freq_str.strip().split(' ')
    vertex_set.add(v1)
    vertex_set.add(v2)
print('enhanced-graph: number of edges: %d, number of vertex: %d, density: %.2f' % (
edge_count, len(vertex_set), edge_count / len(vertex_set)))
