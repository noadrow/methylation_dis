import pyarrow.parquet as pq
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as signal
import numpy as np
from scipy.stats import shapiro
from scipy.stats import kstest
from scipy.stats import anderson
from scipy.stats import skew, kurtosis, jarque_bera
import math
from statsmodels.stats.diagnostic import lilliefors



non_sig = ["cg00009523","cg00009553","cg00009750"]

multipeak = ['cg00147627',
'cg00243527',
'cg00267207',
'cg00308021',
'cg00328916',
'cg00332305',
'cg00366190',
'cg00456685']



def load_new(list=[],path="new_gse_test/"):
    f_df = pd.read_csv(path+"female_df.csv")
    m_df = pd.read_csv(path + "MALE_UNITED_DATA.csv")
    f_info = pd.read_csv(path + "FEMALE_UNITED_DATA_info.csv")
    m_info = pd.read_csv(path + "MALE_UNITED_DATA_info.csv")
    joined_df, joined_info = merge_df(m_df,f_df,f_info,m_info)
    if bool(list):
        filter = set(list).intersection(joined_df.columns.to_list())
        filter = [*filter, ]
        joined_df = joined_df[filter]
    both_df = joined_df.T.merge(joined_info, left_index=True, right_index=True)

    return both_df

def sort_and_clean(df,columns):
    df.index = df[columns]
    df = df.drop(columns=[columns])
    df = df.sort_index()
    df = df.reindex(sorted(df.columns), axis=1)
    print('2. sort_and_clean')
    return df

def merge_df(df_m,df_f,info_f,info_m):

    df_f = df_f.groupby('Unnamed: 0').first()
    df_f = df_f.reindex(sorted(df_f.columns), axis=1)
    df_m = sort_and_clean(df_m,'Unnamed: 0')
    joined_df = df_m.merge(df_f, left_index=True, right_index=True)

    info_m = sort_and_clean(info_m, 'GSM')
    info_m = info_m.drop(columns=[ 'Unnamed: 0'])
    info_f = sort_and_clean(info_f, 'GSM')
    info_f = info_f.drop(columns=['Unnamed: 0'])

    joined_info = info_m.append(info_f)

    print('3. marge_df')
    return [joined_df, joined_info]

def load(list=["NaN"],path = "../data/Healthy_REFBASE.parquet",info_path ='../data/HEALTHY_INFO.parquet'):
    if list==["NaN"]:
        df = pq.read_table(path).to_pandas().iloc[1000:2000]
    else:
        df = pq.read_table(path).to_pandas().loc[list]
    meth_table_info_df = pq.read_table(info_path).to_pandas()
    meth_table_df_both = df.T.merge(meth_table_info_df, left_index=True, right_index=True)
    return meth_table_df_both.T

def load_gse(gse,gse_info_path="../data/healthy_gsm_gse_info.csv",path = "../data/Healthy_REFBASE.parquet",info_path ='../data/HEALTHY_INFO.parquet'):
    df = pq.read_table(path).to_pandas()
    #df = df.iloc[1000:2000]
    meth_table_info_df = pq.read_table(info_path).to_pandas().T.sort_index(axis='index')
    meth_table_df = df.T.sort_index(axis='index')
    meth_table_df_both = meth_table_df.merge(meth_table_info_df.T, left_index=True, right_index=True)
    meth_table_info_df = pd.read_csv(gse_info_path).T.sort_index(axis='index')
    meth_table_info_df.columns = meth_table_info_df.loc['gsms']
    meth_table_df_both_gse = meth_table_df_both.merge(meth_table_info_df.T, left_index=True, right_index=True)

    return meth_table_df_both_gse[meth_table_df_both_gse['gse'] == gse]


def range_count_2(df):
    bin_edges = np.arange(0, 1.00, 0.01)
    hist_list = []
    for i in range(len(df)):
        data = df.iloc[i]
        hist, bin_edges = np.histogram(data, bins=bin_edges)
        hist_list.append(hist)

    d = dict(zip(df.index.to_list(), hist_list))
    new_df = pd.DataFrame.from_dict(d)
    new_df.index = bin_edges.tolist()[1:]
    return new_df.T

def lomb_scargle(df,cg,pi_scal):
    rng = np.random.default_rng()
    y = np.array(df.loc[cg])
    w = np.linspace(0.01, 1, 100000)
    x = np.linspace(0.01, pi_scal*np.pi, 99)

    pgram = signal.lombscargle(x, y, w, normalize=True)

    fig, (ax_t, ax_w) = plt.subplots(2, 1, constrained_layout=True)
    ax_t.plot(x, y, 'b+')
    ax_t.set_xlabel('Time [s]')

    ax_w.plot(w, pgram)
    ax_w.set_xlabel('Angular frequency [rad/s]')
    ax_w.set_ylabel('Normalized amplitude')
    plt.show()


def shapiro_test(df,df_range):
    #results = []
    for cg in df.index[:len(df)]:
        data = df.loc[cg].dropna().to_list()
        plt.bar(np.arange(len(df_range.loc[cg])),df_range.loc[cg])
        result = lilliefors(data)
        # True for normal distribution
        pass_val = "normal" if (result[0] < 0.1) else "not normal"
        plt.title(f"{cg}:{pass_val}")
        plt.text(s=f"satistics: {result[0]}, p_val: {result[1]}", x=-5, y=100, fontsize=11)
        plt.savefig(f"{cg}_{pass_val}_peak_detection.png")
        plt.clf()
    #return results

def lilliefors_test(df,df_range,gse):
    results = []
    for cg in df.index[:len(df)-2]:
        std = df.loc[cg].dropna().std()
        data = df.loc[cg].dropna().to_list()
        result = lilliefors(data)
        # True for normal distribution
        pass_val = "normal" if ((result[0] < 0.1) or (std < 0.1)) else "not normal"
        if pass_val=="not normal":
            plt.title(f"{cg}:{pass_val}, std:{std}, gse:{gse}")
            plt.bar(np.arange(len(df_range.loc[cg])),df_range.loc[cg])
            plt.savefig(f"{cg}_{pass_val}_peak_detection.png")
            plt.clf()
            results.append(cg)
    with open(f'multipeak_cgs_{gse}.txt', 'w') as f:
        f.writelines("%s\n" % cg for cg in results)

    return results

def lilliefors_test_fast(df,gse):
    results,cgs,satistics,stds = [], [], [],[]
    for cg in df.index[:len(df)-2]:
        std = df.loc[cg].dropna().std()
        data = df.loc[cg].dropna().to_list()
        if bool(data):
            result = lilliefors(data)
            pass_val = False if ((result[0] < 0.1) or (std < 0.1)) else True

        else:
            pass_val = False

        # True for normal distribution

        if pass_val:
            results.append(cg)
            stds.append(std)
            satistics.append(result[0])
            cgs.append(cg)

    new_pd = pd.DataFrame({'cg': cgs, 'std': stds, 'satistics': satistics})
    new_pd.to_csv(f'only_multipeak_cgs_{gse}.csv')

    return results

#mult_df = load(multipeak)
#non_sig_df = load(non_sig)
#mult_df_new_df = range_count_2(mult_df)
#non_sig_df_new_df = range_count_2(non_sig_df)
#lomb_scargle(non_sig_df_new_df,non_sig[1],100)
#lomb_scargle(mult_df_new_df,multipeak[1],100)
#shapiro_test(non_sig_df,non_sig_df_new_df)

#mult_p = shapiro_test(mult_df,mult_df_new_df)
#sig_p = shapiro_test(non_sig_df,non_sig_df_new_df)
#print("multiple peaks:" mult_p)
#print("one peaks:" sig_p)

# gse = "GSE111629" #
# gse = "GSE87571" #
#df = load_gse(gse)
#df = df.T.iloc[0:len(df.columns)-4]
#df_r = range_count_2(df)
#lilliefors_test(df,df_r,gse)

#gse = "GSE87571" #
#df = load_gse(gse)
#df = df.T.iloc[0:len(df.columns)-4]
#lilliefors_test_fast(df,gse)

#gse = "GSE40239" #426
#gse = "GSE42861" #335

# gse = "GSE42861"
# gse = "GSE40239"
# df = load_new()
# df.index = df['GSE']
# df = df.T.iloc[0:len(df.columns)-2]
# lilliefors_test_fast(df.iloc[:,df.columns==gse],gse)
#gse="GSE111629"
#gse = "GSE87571"
gse = "GSE40279"
path=f"../data/new_gse_test/{gse}_normalized_celltype.csv"
df = pd.read_csv(path,index_col=0)
lilliefors_test_fast(df,gse)

"""
female_info  = pd.read_csv("new_gse_test/FEMALE_UNITED_DATA_info.csv")['GSM']
male_info = pd.read_csv("new_gse_test/MALE_UNITED_DATA_info.csv")['GSM']

fm_info = set(df.columns).intersection(female_info)
ml_info = set(df.columns).intersection(male_info)

df_fm = df[fm_info]
df_ml = df[ml_info]
s
"""

#lilliefors_test_fast(df,f"{gse}_all")
#lilliefors_test_fast(df_ml,f"{gse}_ml")