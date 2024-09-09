import pandas as pd
from aquarel import load_theme
from matplotlib import pyplot as plt

base_dir = './'

all_themes = ['scientific', 'boxy_dark', 'boxy_light', 'umbra_light']
theme = load_theme(all_themes[3])
theme.apply()

colors = ['#%02x%02x%02x' % (n[0], n[1], n[2]) for n in
          [(224, 131, 103), (140, 197, 140), (238, 163, 165), (140, 150, 236), (225, 128, 99), (0, 169, 217), (4, 173, 149), (91, 155, 213), (217, 152, 52)]]
print(colors)

all_databases = ['ACL', 'ACM', 'Arxiv', 'DBLP', 'MAG', 'OAG', 'PMC', 'PubMed', 'S2']
all_methods = ['D', 'T', 'D+T']

df_author_pub_cnt = pd.read_csv('author_pub_cnt.csv')
df_institution_pub_cnt = pd.read_csv('institution_pub_cnt.csv')
df_db_fos = pd.read_csv('db_fos.csv')

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))
ax = axes[0]
ax.plot(df_author_pub_cnt['num_published_papers'].tolist(), color=colors[0], alpha=1.0, linewidth=3.5)
ax.set_yscale('log')
top_author_list = '\n'.join(
    [str(i + 1) + '. ' + n[0] + ' (ID:' + str(n[1]) + ')' for i, n in
     enumerate(df_author_pub_cnt[['author_name', 'AuthorId']][:10].values)])
print(top_author_list)
ax.plot(0, df_author_pub_cnt['num_published_papers'][0], marker='o', alpha=0.4, ms=10, mfc=(1., 0., 0., 0.5), mec='None')
ax.annotate(top_author_list, xy=(0, df_author_pub_cnt['num_published_papers'][0]),
            xytext=(6000, 7),
            arrowprops=dict(arrowstyle="->", facecolor='red', color='grey'),
            bbox=dict(facecolor='none', edgecolor='grey'),
            fontsize=11
            )
ax.set_title('(a) Author publication count in ACL', fontsize=16)
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)
ax.grid(False)

ax = axes[1]
ax.plot(df_institution_pub_cnt['num_published_papers'].tolist(), color=colors[0], alpha=1.0, linewidth=3.5)
ax.set_yscale('log')
ax.set_yscale('log')
top_institution_list = '\n'.join([str(i + 1) + '. ' + n[0] + ' (ID:' + str(n[1]) + ')' for i, n in
                                  enumerate(df_institution_pub_cnt[['DisplayName', 'AffiliationId']][:10].values)])
print(top_institution_list)
ax.plot(0, df_institution_pub_cnt['num_published_papers'][0], marker='o', alpha=0.4, ms=10, mfc=(1., 0., 0., 0.5), mec='None')
ax.annotate(top_institution_list, xy=(0, df_institution_pub_cnt['num_published_papers'][0]),
            xytext=(200, 14),
            arrowprops=dict(arrowstyle="->", facecolor='red', color='grey'),
            bbox=dict(facecolor='none', edgecolor='grey'),
            fontsize=11
            )
ax.set_title('(b) Institution publication count in ACL', fontsize=16)
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)
ax.grid(False)

ax = axes[2]
fos_cnt = df_db_fos[['fos', 'cnt']].values
fos = [n[0] for n in fos_cnt]
weight = [n[1] for n in fos_cnt]
weight = [round(n * 100, 2) / sum(weight) for n in weight]
ax.bar(fos, weight, width=0.5, color=colors[0], alpha=1.0)
ax.bar_label(ax.containers[0], label_type='edge', fmt='%.1f', fontsize=11)
ax.set_ylim([0, 65])
ax.set_yticklabels([str(int(x)) + '%' for x in ax.get_yticks()])  # y 轴加上百分号
ax.set_xticklabels(fos, rotation=20, ha='right')
ax.set_title('(c) Domain coverage of arXiv', fontsize=16)
ax.tick_params(axis='x', labelsize=11)
ax.tick_params(axis='y', labelsize=13)
ax.grid(False)

plt.tight_layout()
plt.savefig(base_dir + '/figs/application-for-bibliometrics.png', dpi=600)
plt.savefig(base_dir + '/figs/application-for-bibliometrics.pdf')

plt.show()
