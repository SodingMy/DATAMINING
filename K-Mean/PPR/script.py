import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import numpy as np

existing_df = pd.read_csv(
    'household-public.csv', 
    index_col = 0, 
    thousands  = ',')
existing_df.index.names = ['ID']
existing_df.columns.names = ['column_name']

existing_df.head()

"""
print existing_df.boxplot(column='Tinggal di PPR semenjak')
plt.show()
print existing_df.boxplot(column='Bagi responden yang tinggal di PPR kurang daripada 12 bulan, sila nyatakan tempoh dalam bulan:')
plt.show()
"""

del existing_df['Timestamp']
del existing_df['Telefon']
del existing_df['Jumlah tanggungan bil telefon, sila nyatakan:']
del existing_df['Tarikh survey ini di isikan oleh enumerator atau responden']

existing_df.columns = ['c_1', 'c_2', 'c_3', 'c_4', 'c_5', 'c_6', 'c_7', 'c_8', 'c_9', 'c_10', 'c_11', 'c_12', 'c_13', 'c_14', 'c_15', 'c_16', 'c_17', 'c_18', 'c_19', 'c_20', 'c_21', 'c_22', 'c_23', 'c_24', 'c_25', 'c_26', 'c_27', 'c_28', 'c_29', 'c_30']

existing_df['c_2'].fillna(existing_df['c_2'].mean(), inplace=True)
existing_df['c_3'].fillna(existing_df['c_3'].mean(), inplace=True)


existing_df['c_4'] = pd.factorize(existing_df.c_4)[0]
existing_df['c_5'] = pd.factorize(existing_df.c_5)[0]

existing_df['c_6'] = pd.factorize(existing_df.c_6)[0]
existing_df['c_7'] = pd.factorize(existing_df.c_7)[0]
existing_df['c_8'] = pd.factorize(existing_df.c_8)[0]
existing_df['c_9'] = pd.factorize(existing_df.c_9)[0]
existing_df['c_10'] = pd.factorize(existing_df.c_10)[0]
existing_df['c_11'] = pd.factorize(existing_df.c_11)[0]
existing_df['c_12'] = pd.factorize(existing_df.c_12)[0]
existing_df['c_13'] = pd.factorize(existing_df.c_13)[0]
existing_df['c_14'] = pd.factorize(existing_df.c_14)[0]
existing_df['c_15'] = pd.factorize(existing_df.c_15)[0]
existing_df['c_16'] = pd.factorize(existing_df.c_16)[0]
existing_df['c_18'] = pd.factorize(existing_df.c_18)[0]
existing_df['c_19'] = pd.factorize(existing_df.c_19)[0]
existing_df['c_20'] = pd.factorize(existing_df.c_20)[0]
existing_df['c_21'] = pd.factorize(existing_df.c_21)[0]
existing_df['c_22'] = pd.factorize(existing_df.c_22)[0]
existing_df['c_23'] = pd.factorize(existing_df.c_23)[0]
existing_df['c_24'] = pd.factorize(existing_df.c_24)[0]
existing_df['c_25'] = pd.factorize(existing_df.c_25)[0]
existing_df['c_26'] = pd.factorize(existing_df.c_26)[0]
existing_df['c_27'] = pd.factorize(existing_df.c_27)[0]
existing_df['c_28'] = pd.factorize(existing_df.c_28)[0]
existing_df['c_29'] = pd.factorize(existing_df.c_29)[0]
existing_df['c_30'] = pd.factorize(existing_df.c_30)[0]

print existing_df.head(10)

print existing_df.apply(lambda x: sum(x.isnull()),axis=0) 

"""
pca = PCA(n_components=2)
pca.fit(existing_df)

existing_2d = pca.transform(existing_df)

existing_df_2d = pd.DataFrame(existing_2d)
existing_df_2d.index = existing_df.index
existing_df_2d.columns = ['PC1','PC2']
existing_df_2d.head()

pca.explained_variance_ratio_

ax = existing_df_2d.plot(kind='scatter', x='PC2', y='PC1', figsize=(16,8))

for i, country in enumerate(existing_df.index):
    ax.annotate(
        country, 
        (existing_df_2d.iloc[i].PC2, existing_df_2d.iloc[i].PC1)
    )


existing_df_2d['a_login_mean'] = pd.Series(existing_df.mean(axis=1), index=existing_df_2d.index)
a_login_mean_max = existing_df_2d['a_login_mean'].max()
a_login_mean_min = existing_df_2d['a_login_mean'].min()
a_login_mean_scaled = (existing_df_2d.a_login_mean-a_login_mean_min) / a_login_mean_max
existing_df_2d['a_login_mean_scaled'] = pd.Series(
        a_login_mean_scaled, 
        index=existing_df_2d.index) 
existing_df_2d.head()

existing_df_2d.plot(
    kind='scatter', 
    x='PC2', 
    y='PC1', 
    s=existing_df_2d['a_login_mean_scaled']*100, 
    figsize=(16,8))

existing_df_2d['a_login_sum'] = pd.Series(
    existing_df.sum(axis=1), 
    index=existing_df_2d.index)
a_login_sum_max = existing_df_2d['a_login_sum'].max()
a_login_sum_min = existing_df_2d['a_login_sum'].min()
a_login_sum_scaled = (existing_df_2d.a_login_sum-a_login_sum_min) / a_login_sum_max
existing_df_2d['a_login_sum_scaled'] = pd.Series(
        a_login_sum_scaled, 
        index=existing_df_2d.index)
existing_df_2d.plot(
    kind='scatter', 
    x='PC2', y='PC1', 
    s=existing_df_2d['a_login_sum_scaled']*100, 
    figsize=(16,8))


existing_df_2d['a_login_change'] = pd.Series(
    existing_df['2007']-existing_df['1990'], 
    index=existing_df_2d.index)
country_change_max = existing_df_2d['country_change'].max()
country_change_min = existing_df_2d['country_change'].min()
country_change_scaled = 
    (existing_df_2d.country_change - country_change_min) / country_change_max
existing_df_2d['country_change_scaled'] = pd.Series(
        country_change_scaled, 
        index=existing_df_2d.index)
existing_df_2d[['country_change','country_change_scaled']].head()

existing_df_2d.plot(
    kind='scatter', 
    x='PC2', y='PC1', 
    s=existing_df_2d['country_change_scaled']*100, 
    figsize=(16,8))


kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit(existing_df)

existing_df_2d['cluster'] = pd.Series(clusters.labels_, index=existing_df_2d.index)

existing_df_2d.plot(
        kind='scatter',
        x='PC2',y='PC1',
        c=existing_df_2d.cluster.astype(np.float), 
        figsize=(16,8))

existing_df_2d['cluster'].to_dict()

plt.show(block=True)
"""
