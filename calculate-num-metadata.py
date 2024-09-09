from collections import Counter

import numpy as np
import pandas as pd

based_dir = './'
df = pd.read_excel(based_dir + '/docs/analysis.xlsx', sheet_name='metadata-statistic')
df = df[df['Normalized Name'].notnull()]
df = df.drop_duplicates(subset=['database', 'Normalized Name'], keep=False, inplace=False)
print(df)

for db, df_db in df.groupby('database'):
    metadata_list = df_db['Normalized Name'].values.tolist()
    assert len(metadata_list) == len(set(metadata_list))
    print(db.upper(), metadata_list)

all_metadata_list = df['Normalized Name'].values.tolist()
metadata_freq_count = Counter(all_metadata_list)

print('*' * 100)
print(len(metadata_freq_count), len(all_metadata_list))
metadata_freq_count = [name + '\t' + str(freq) for (name, freq) in
                       sorted(metadata_freq_count.items(), key=lambda x: x[1], reverse=True)]

num_cols = 3
num_row = len(metadata_freq_count) // num_cols
if num_row * num_cols != len(metadata_freq_count):
    num_row += 1

arr = np.empty(shape=(num_row, num_cols), dtype=object)

for i in range(len(metadata_freq_count)):
    arr[i % num_row][i // num_row] = metadata_freq_count[i]

for n in arr:
    print('\t'.join([m if m is not None else '' for m in n]))
