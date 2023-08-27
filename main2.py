import pyarrow.parquet as pq
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

def read_cgs(file_path):
    with open(file_path, "r") as file:
        lines = [line.strip() for line in file]
    return lines

def sep_by(df,column):
    values = list(df[column].unique())
    dfs = []
    for value in values:
        mask = df[column] == value
        filtered_df = df[mask]
        dfs.append(filtered_df)
    d = dict(zip(values, dfs))
    return d

def calc_vec(df,res):
    d_gender = sep_by(df.T,'gender')
    genders = []
    ages = []
    for gender in d_gender:
        genders.append(gender)
        ages.append(sep_by(d_gender[gender], 'age'))
    d = dict(zip(genders, ages))

    genders = []
    ages = []
    cgs = []
    vectors = []
    for gender in d:
        gender_d = d[gender]
        for age in gender_d:
            age_d = gender_d[age]
            cg_list = list(age_d.columns[:-2])
            for cg in cg_list:
                cgs.append(cg)
                genders.append(gender)
                ages.append(age)

                total = len(age_d[cg].index)
                vec = age_d[cg].groupby(pd.cut(age_d[cg],np.arange(0, 1, res))).count()
                vec = vec.values * (1 / total)
                #vec = np.convolve(vec,vec)
                vectors.append(vec)

    data = pd.DataFrame({
            'gender': genders,
            'age': ages,
            'cg': cgs,
            'ratio_vector': vectors})
    return data

def calc_stats(df):
    d_gender = sep_by(df.T,'gender')
    genders = []
    ages = []
    for gender in d_gender:
        genders.append(gender)
        ages.append(sep_by(d_gender[gender], 'age'))
    d = dict(zip(genders, ages))

    genders = []
    ages = []
    cgs = []
    vectors = []
    for gender in d:
        gender_d = d[gender]
        for age in gender_d:
            age_d = gender_d[age]
            cg_list = list(age_d.columns[:-2])
            for cg in cg_list:
                cgs.append(cg)
                genders.append(gender)
                ages.append(age)

                total = len(age_d[cg].index)
                mean = age_d[cg].mean()
                std = age_d[cg].std()
                vectors.append((mean,std))

    data = pd.DataFrame({
            'gender': genders,
            'age': ages,
            'cg': cgs,
            'mean_std_vec': vectors})
    return data

def graph_cg_to_beta_dist(df,batch,chunk_size = 10):
        for i in range(0,df.shape[0]-2):
            cg = df.index[i]
            ages = df.loc['age']
            genders = df.loc['gender']
            new_df = pd.DataFrame({cg:df.iloc[i], 'age':ages, 'gender':genders})
            fig = sns.catplot(data=new_df, x="age", y=cg, hue="gender", jitter=False).fig
            fig.savefig(f'{cg}.png')

            #new_df = new_df.astype({cg: float})
            #fig = sns.catplot(data=new_df, x="age", y=cg, hue="gender", kind="violin", split=True).fig


def Extract(lst,i):
    return [item[i] for item in lst]

def PCA_calc(data):
    X = data['ratio_vector'].to_numpy()
    list = []
    for i in range(0, len(X)):
        list.append(X[i].tolist())
    pca = PCA(n_components=2)
    Xt = pca.fit_transform(list)
    data['PCA1'] = Extract(Xt, 0)
    data['PCA2'] = Extract(Xt, 1)
    return data

def stats_PCA_match(data,):
    X = data['mean_std_vec'].to_numpy()
    data['PCA1'] = Extract(X, 0)
    data['PCA2'] = Extract(X, 1)
    return data

def Umap_calc(data):
    import umap
    from sklearn.preprocessing import StandardScaler
    reducer = umap.UMAP()
    X = list(data['ratio_vector'])
    scaled_X = StandardScaler().fit_transform(X)
    Xt = reducer.fit_transform(scaled_X)
    data['PCA1'] = Extract(Xt, 0)
    data['PCA2'] = Extract(Xt, 1)
    return data

def Tsne_calc(data):
    from sklearn.manifold import TSNE
    X = np.array(list(data['ratio_vector']))
    Xt = TSNE(n_components=2, learning_rate='auto',init = 'random', perplexity = 3).fit_transform(X)
    data['PCA1'] = Extract(Xt, 0)
    data['PCA2'] = Extract(Xt, 1)
    return data

def mean_std_total(df):
    means = []
    stds = []
    cgs = []
    cgs_dfs = df.groupby('cg')
    for name,g in cgs_dfs:
        cgs.append(name)
        mean_list = Extract(g['mean_std_vec'], 0)
        std_list = Extract(g['mean_std_vec'], 1)
        means.append([np.nanmean(np.array(mean_list)),np.nanstd(np.array(mean_list))])
        stds.append([np.nanmean(np.array(std_list)),np.nanstd(np.array(std_list))])

    new_df = pd.DataFrame({'cg':cgs,
                           'mean': means,
                           'std': stds})
    return new_df

def arrange_stats(dfs,types):
    cgs, groups, mean_means,mean_stds, std_means, std_stds = [],[],[],[],[],[]

    for (df,type) in zip(dfs,types):
        cgs.extend(df['cg'])
        groups.extend(3 * [type])
        mean_means.extend(Extract(df['mean'],0))
        mean_stds.extend(Extract(df['mean'],1))
        std_means.extend(Extract(df['std'],0))
        std_stds.extend(Extract(df['std'],1))

    new_pd = pd.DataFrame({'cg': cgs,
                           'group': groups,
                            'mean_mean': mean_means,
                            'mean_std': mean_stds,
                            'std_mean': std_means,
                            'std_std': std_stds})
    return new_pd

def graph_PCA(df,batch):
    df_g = df.groupby('cg')
    for cg,g in df_g:
        fig = sns.catplot(data=g, x="age", y="PCA1", hue="gender", jitter=False).fig
        fig.savefig(f'{batch}_{cg}_PCA2.png')
        fig = sns.catplot(data=g, x="age", y="PCA2", hue="gender", jitter=False).fig
        fig.savefig(f'{batch}_{cg}_PCA1.png')

def scatter_3d(df,batch,calc):
    df_g = df.groupby('cg')
    for cg, g in df_g:
        plot_3d(g,cg,batch,calc)

def plot_3d(g,cg,batch,calc):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(g['age'], g['PCA1'], g['PCA2'])
    ax.set_xlabel('age')
    ax.set_ylabel(f'{calc}1')
    ax.set_zlabel(f'{calc}2')
    plt.show()

def barplot_range_count(df,group):
    cgs = df.index.to_list()[:3]
    for cg in cgs:
        working_df = df.loc[cg]
        keyList = pd.cut(working_df.values,np.arange(0, 1, 0.01)).categories.values
        new_pd = pd.DataFrame({
            'range': pd.cut(working_df.values,np.arange(0, 1, 0.01)),
            'val': working_df,
            'index': working_df.index,
            'counter': [1] * len(working_df)
        })
        range_count = new_pd.groupby('range')['counter'].count()

        ax = range_count.plot.bar(rot=90,figsize=(15,10))
        fig = ax.figure
        fig.savefig(f"{cg}_{group}_barh.png")
        plt.clf()
        plt.close(fig)

def graph_total_dis(df,group):

    cgs = df.index.to_list()[:3]
    beta_vals,cg_list,genders = [],[],[]

    for cg in cgs:
        beta_vals.extend(df.loc[cg].values)
        cg_list.extend(len(df.loc[cg]) * [cg])
        genders.extend(df.loc["gender"])
    new_df = pd.DataFrame({'cg':cg_list,'beta':beta_vals,'gender':genders})
    sns.catplot(data=new_df, x="cg", y='beta', hue="gender", jitter=False).fig.savefig(f"{group}_total_dis.png")

gse = "GSE111629"
file_path = f"../new_gse_test/{gse}_normalized_celltype.csv"
df = pd.read_csv(file_path)

info_path = '../new_gse_test/GSE111629_info.csv'
meth_table_info_df = pd.read_csv(info_path).sort_index(axis='index')
meth_table_info_df.index = meth_table_info_df['GSM']
df.index = df['Unnamed: 0']
meth_table_df = df.T.sort_index(axis='index')
meth_table_df_both = meth_table_df.merge(meth_table_info_df,left_index=True, right_index=True)

file_path = f"../trash/multipeak_cgs_GSE111629.no_chrx.only_cgs.txt"

# cgs = read_cgs(file_path)
# bottom_10_lilifores
# cgs = ['cg06532184','cg07185843','cg11041161']
# top_10_lilifores
# cgs = ['cg07629776','cg21238284','cg22132193']
# IGF2BP1
# cgs = ['cg22947322']
# GFPT2
# cgs = ['Cg01758122']

#file_path=f"../methylToSNP.txt"
#cgs = read_cgs(file_path)
#cgs = ['cg04913934', 'cg08314949', 'cg12078154','cg17375167','cg03760951']
cgs = ['cg18846074','cg10862468','cg11445109','cg13315147','cg18984983','cg19469447','cg23400446','cg24530264']
fit_cgs = list(set(df.index).intersection(cgs))
fit_cgs.extend(['gender','age'])
df_cg = meth_table_df_both[fit_cgs]

graph_cg_to_beta_dist(df_cg.T,"GSE87571_nochrx")