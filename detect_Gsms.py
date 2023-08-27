import pyarrow.parquet as pq
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load(list=["NaN"],path = "../data/Healthy_REFBASE.parquet",info_path ='../data/HEALTHY_INFO.parquet'):
    if list==["NaN"]:
        df = pq.read_table(path).to_pandas().iloc[1000:2000]
    else:
        df = pq.read_table(path).to_pandas().loc[list]
    meth_table_info_df = pq.read_table(info_path).to_pandas()
    meth_table_df_both = df.T.merge(meth_table_info_df, left_index=True, right_index=True)
    return meth_table_df_both.T

def load_gse(gse="",gse_info_path="../data/healthy_gsm_gse_info.csv",path = "../data/Healthy_REFBASE.parquet",info_path ='../data/HEALTHY_INFO.parquet'):
    df = pq.read_table(path).to_pandas()
    #df = df.iloc[1000:2000]
    meth_table_info_df = pq.read_table(info_path).to_pandas().T.sort_index(axis='index')
    meth_table_df = df.T.sort_index(axis='index')
    meth_table_df_both = meth_table_df.merge(meth_table_info_df.T, left_index=True, right_index=True)
    meth_table_info_df = pd.read_csv(gse_info_path).T.sort_index(axis='index')
    meth_table_info_df.columns = meth_table_info_df.loc['gsms']
    meth_table_df_both_gse = meth_table_df_both.merge(meth_table_info_df.T, left_index=True, right_index=True)
    if gse == "":
        return meth_table_df_both_gse
    else:
        return meth_table_df_both_gse[meth_table_df_both_gse['gse'] == gse]


def read_cgs(file_path):
    with open(file_path, "r") as file:
        lines = [line.strip() for line in file]
    return lines

def graph_cg_to_beta_dist_gse(df, batch, color_list):
    # for i in range(0, df.shape[0] -2):
    # for i in range(0, df.shape[0]):
    # cg = df.columns.to_list()[i]
    #ages = df.loc['age']
    gsms = df.index.to_list()
    new_df = pd.DataFrame({'cg': df, 'gsms': gsms})
    # new_df = new_df.sort_values(by=cg)
    new_df = new_df.sort_values(by='cg')
    # new_df['is_middle'] = new_df['gsms'].isin(color_list)

    # fig = sns.catplot(data=new_df, x="gsms", y=cg , hue="is_middle", jitter=False).fig
    fig = sns.catplot(data=new_df, x="gsms", y="cg", jitter=False).fig
    plt.yticks(np.arange(0, 1, 0.05))
    fig.savefig(f'{cg}_{batch}.png')
    plt.clf()
    plt.close(fig)

def filter_gsm(df, cg, min, max):
    results = df[df[cg] > min][df[cg] < max].index
    with open(f'middlepeak_gsms_{cg}.txt', 'w') as f:
        f.writelines("%s\n" % gsm for gsm in results)

def test_filter_gsm(df,cg):
    df_cg = df[cg].sort_values()

    if(pd.api.types.is_float_dtype(df_cg)):
        group1 = df_cg[df_cg < np.float32(0.2)]
        group2 = df_cg[(df_cg >= np.float32(0.2)) & (df_cg < np.float32(0.6))]
        group3 = df_cg[df_cg >= np.float32(0.6)]
        return [group1.index.to_list(), group2.index.to_list(), group3.index.to_list()]
    else:
        return []


def cut_off(df,cgs):
    results_top = []
    results_middle = []
    results_down = []
    for cg in cgs:
        results = test_filter_gsm(df,cg)
        if bool(results):
            #results_top.extend(results[0])
            results_middle.extend(results[1])
            #results_down.extend(results[2])

    #top_sum = pd.DataFrame({'top':results_top}).groupby('top')['top'].count().sort_values(ascending=False)
    middle_sum = pd.DataFrame({'middle':results_middle}).groupby('middle')['middle'].count().sort_values(ascending=False)[0:100]
    #down_sum = pd.DataFrame({'down':results_down}).groupby('down')['down'].count().sort_values(ascending=False)
    with open(f'middle_sum_gsms.txt', 'x') as f:
        f.writelines("%s\n" % gsm for gsm in middle_sum.index)


# cgs = read_cgs("intersected_GSE87571_GSE111629.txt")
# cgs.extend(['gender','age','gse','gsms'])
# df = load_gse("GSE111629")[cgs] #GSE87571

#cgs_filter = ["cg00072288","cg00113623","cg00167248","cg00348031","cg00377727","cg00197266","cg00051154", "cg00087746", "cg00095677", "cg00112256"]

#gsms = read_cgs(f"middlepeak_gsms_{cgs_filter[1]}.txt")
#cut_off(df,cgs)
#graph_cg_to_beta_dist_gse(df,f"check_middle_{cgs_filter[8]}",gsms)

'''
#"cg00072288"
filter_gsm(df,"cg00072288",0.3,0.5)

# "cg00051154"
filter_gsm(df,"cg00051154",0.35,0.65)

# "cg00087746"
filter_gsm(df,"cg00087746",0.0,0.2)

# "cg00095677"
filter_gsm(df,"cg00095677",0.35,0.5)

# "cg00112256"
filter_gsm(df,"cg00112256",0.35,0.6)

# "cg00113623"
filter_gsm(df,"cg00113623",0.4,0.6)

# "cg00167248"
filter_gsm(df,"cg00167248",0.25,0.5)

# "cg00348031"
filter_gsm(df,"cg00348031",0.0,0.2)

# "cg00377727"
filter_gsm(df,"cg00377727",0.55,0.8)

# "cg00197266"
filter_gsm(df,"cg00197266",0.3,0.55)

'''

# gse="GSE111629"
gse="GSE87571"
path=f"new_gse_test/{gse}_normalized_celltype.csv"
df = pd.read_csv(path)
df.index = df['Unnamed: 0']
df = df.drop(columns=['Unnamed: 0'])
# cgs = read_cgs(f'AFTER_NORM/multipeak_cgs_{gse}.txt')
file_path=f"AFTER_NORM/{gse}_lilifores.txt"
cgs = read_cgs(file_path)
for cg in cgs:
    graph_cg_to_beta_dist_gse(df.loc[cg], gse, [])
