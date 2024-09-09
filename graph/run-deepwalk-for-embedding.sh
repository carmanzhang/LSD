#!/bin/bash

# Note refer to the github repo: DeepWalk-dgl
DIR=/home/zhangli/mydisk-3t/repo/scholarly-bigdata/cached/graph-embedding

python3 deepwalk.py --net_file $DIR/original-pmc-hetero-as-homo-graph-2023-12-24-15-05.tsv --emb_file emb.txt --adam --mix --lr 0.2 --mix --num_threads 15 --batch_size 100 --negative 3 --walk_length 30 --print_interval 1000 --num_walks 5 --window_size 5 --dim 64 --map_file $DIR/original-pmc-hetero-as-homo-graph-2023-12-24-15-05.tsv.idmap --emb_file $DIR/original-pmc-hetero-as-homo-graph-2023-12-24-15-05.tsv.emb
#Training used time: 5736.04s
#Total used time: 6046.13
python3 deepwalk.py --net_file $DIR/enhanced-pmc-hetero-as-homo-graph-2023-12-24-15-05.tsv --emb_file emb.txt --adam --mix --lr 0.2 --mix --num_threads 15 --batch_size 100 --negative 3 --walk_length 30 --print_interval 1000 --num_walks 5 --window_size 5 --dim 64 --map_file $DIR/enhanced-pmc-hetero-as-homo-graph-2023-12-24-15-05.tsv.idmap --emb_file $DIR/enhanced-pmc-hetero-as-homo-graph-2023-12-24-15-05.tsv.emb
