import matplotlib.pyplot as plt
import pandas as pd
from aquarel import load_theme

all_themes = ['scientific', 'boxy_dark', 'boxy_light', 'umbra_light']
theme = load_theme(all_themes[3])
theme.apply()

based_dir = './'

colors = ['#%02x%02x%02x' % (n[0], n[1], n[2]) for n in [(4, 173, 149), (238, 163, 165), (140, 150, 236), (225, 128, 99), (0, 169, 217), (91, 155, 213), (217, 152, 52)]]

database_num_papers = {k: v for k, v in pd.read_excel(based_dir + '/docs/analysis.xlsx',
                                                      sheet_name='scholarly-databases-statistics')[['Database', '\# Articles']].values}
print(database_num_papers)

df = pd.read_excel(based_dir + '/docs/analysis.xlsx', sheet_name='linking1')
print(df)

all_databases = ['ACL', 'ACM', 'Arxiv', 'DBLP', 'MAG', 'OAG', 'PMC', 'PubMed', 'S2']

df_D = df[df['Method'] == 'D']
df_D['database'] = all_databases
df_T = df[df['Method'] == 'T']
df_T['database'] = all_databases
df_DT = df[df['Method'] == 'D+T']
df_DT['database'] = all_databases

db_2_db_linking_map = {}
for col in all_databases:
    num_db_linkages_using_DOI = df_D[['database', col]].values
    num_db_linkages_using_Title = df_T[['database', col]].values
    num_db_linkages_using_DOITitle = df_DT[['database', col]].values
    assert [n[0] for n in num_db_linkages_using_DOI] == [n[0] for n in num_db_linkages_using_Title] == [n[0] for n in
                                                                                                        num_db_linkages_using_DOITitle]

    tmp_dict = {'-'.join(sorted([col, db_name])): [v1, v2, v3] for (db_name, v1, v2, v3) in zip(
        [n[0] for n in num_db_linkages_using_DOI],  # DB name
        [n[1] for n in num_db_linkages_using_DOI],
        [n[1] for n in num_db_linkages_using_Title],
        [n[1] for n in num_db_linkages_using_DOITitle],
    )}
    db_2_db_linking_map.update(tmp_dict)

print(db_2_db_linking_map)

num_databases = len(all_databases)

figure, axes = plt.subplots(figsize=(9.9, 11.3),
                            nrows=num_databases - 1, ncols=num_databases - 1,
                            sharex=True, sharey=True)
num_databases = len(all_databases)
for i, db1 in enumerate(all_databases[1:]):
    for j, db2 in enumerate(all_databases[:-1]):

        if i < j:
            figure.delaxes(axes[i][j])
            continue

        db1_num_instances = database_num_papers[db1]
        db2_num_instances = database_num_papers[db2]
        db_pair_min_num_instances = min(db1_num_instances, db2_num_instances)
        db_pair_key = '-'.join(sorted([db1, db2]))
        db_pair_num_linkages_using_various_methods = db_2_db_linking_map[db_pair_key]
        db_pair_pctg_linkages_using_various_methods = [round(n * 100 / db_pair_min_num_instances, 2) for n in
                                                       db_pair_num_linkages_using_various_methods]

        ax = axes[i][j]
        # bars = ax.bar(['D', 'T', 'D+T'], db_pair_pctg_linkages_using_various_methods, color='#6A8677', alpha=0.5, width=0.5)
        bars = ax.bar(['D', 'T', 'D+T'], db_pair_pctg_linkages_using_various_methods, color='green', alpha=0.45, width=0.5)
        ax.bar_label(bars, ['         ' + str(n) for n in db_pair_pctg_linkages_using_various_methods], label_type='center', rotation=90, padding=0,
                     color='black', alpha=1.0,
                     # Note 加粗显示
                     # weight='bold'
                     )
        ax.grid(False)
        # ax.bar_label(rects1, labels=["ABC","ABC","ABC","ABC","ABC"], padding=3, label_type='center')
        if i == num_databases - 2:
            ax.set_xlabel(db2, weight='bold')

        if j == 0:
            ax.set_ylabel(db1, weight='bold')

figure.patch.set_linewidth(2)
figure.patch.set_edgecolor('black')

plt.tight_layout()
plt.savefig(based_dir + '/figs/databases-num-linkages.png', dpi=1000)
plt.savefig(based_dir + '/figs/databases-num-linkages.pdf')
plt.show()
