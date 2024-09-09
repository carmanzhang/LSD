import numpy as np
import pandas as pd

from myio.data_reader import DBReader

all_databases = ['ACL', 'ACM', 'Arxiv', 'DBLP', 'MAG', 'OAG', 'PMC', 'PubMed', 'S2', 'ORCID', 'Crossref']

results = []
num_db = len(all_databases)
for i in range(0, num_db - 1, 1):
    for j in range(i + 1, num_db, 1):
        db1, db2 = all_databases[i], all_databases[j]
        df_res = DBReader.tcp_model_cached_read(sql_template % (db2, db1))
        print(df_res)
        df_res.to_csv('scholarly-databases-number-linkages-bypair.csv', mode='a', header=(i == 0 and j == 1), index=False)
        results.append(df_res)

df_res = pd.read_csv('scholarly-databases-number-linkages-bypair.csv')
print(df_res)
db_pair_linkages_map = {(n[0] + '-' + n[1] if n[0] < n[1] else n[1] + '-' + n[0]): np.array([n[2], n[3], n[4]]) for n in
                        df_res[['database1', 'database2', 'num_D_linkages', 'num_T_linkages', 'num_DT_linkages']].values.tolist()}

db_list = ['ACL', 'ACM', 'Arxiv', 'DBLP', 'MAG', 'OAG', 'PMC', 'PubMed', 'S2']

num_db = len(db_list)
data = np.zeros(shape=(num_db, num_db, 3))
for i in range(num_db):
    for j in range(num_db):
        key = db_list[i] + '-' + db_list[j] if db_list[i] < db_list[j] else db_list[j] + '-' + db_list[i]
        if key in db_pair_linkages_map:
            data[i][j] = db_pair_linkages_map[key]
print(data)

df = pd.DataFrame(data[:, :, 0].tolist() + data[:, :, 1].tolist() + data[:, :, 2].tolist(),
                  index=[n + '-method1' for n in db_list] + [n + '-method2' for n in db_list] + [n + '-method3' for n in db_list],
                  columns=db_list)
df['Method'] = ['D'] * num_db + ['T'] * num_db + ['D+T'] * num_db
df = df.sort_index()
df = df[['Method'] + db_list]
df.index = sum([['\midrule \multirow{3}{*}{%s}' % n, '', ''] for n in db_list], [])
df.to_csv('scholarly-databases-number-linkages-byarray.csv')
