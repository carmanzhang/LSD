from itertools import groupby

import pandas as pd

df = pd.read_csv('database_normalized_metadata.tsv', sep='\t')

df = df.dropna()
print(df)

df['database_metadata'] = df[['database', 'Normalized Name']].apply(lambda x: '_'.join(x), axis=1)
print(df)
print(set(df['database_metadata'].values))
df = df.sort_values(by=['database_metadata']).drop_duplicates('database_metadata', keep='last')
print(df)

unique_metadata = set(df['Normalized Name'].values)
print('unique_metadata: ', unique_metadata, len(unique_metadata))

unique_databases = set(df['database'].values)
print('unique_databases: ', unique_databases, len(unique_databases))

# database_metadata = df[['database', 'Normalized Name']].values
database_metadata = df[['Normalized Name', 'database']].values

res = [[k, list(g)] for k, g in groupby(sorted(database_metadata, key=lambda x: x[0]), key=lambda x: x[0])]
res = sorted(res, key=lambda x: len(x[1]), reverse=True)
res = [[a, len(b)] for a, b in res]
# res = [a for a, b in res]
print(res)

databse_metadata_frequency = pd.DataFrame(res, columns=['Entity Type (Metadata)', 'Frequency'], index=None).sort_values(
    by=['Frequency'],
    ascending=False)
databse_metadata_frequency.to_csv('database-entity-frequency.csv', sep=',')
