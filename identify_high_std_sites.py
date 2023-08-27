import pyarrow.parquet as pq
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import signal
from sklearn.mixture import GaussianMixture

from tkinter import Tk
from tkinter import filedialog
from tkinter.filedialog import askopenfilename

group_1 = ["cg06657917",
		"cg00169354",
		"cg17172308",
		"cg03343571",
		"cg08164151"]

group_2 = ["cg13106476",
		"cg15635302",
		"cg23544472"]

group_3 = ["cg06341731",
		"cg09918751",
		"cg13857646",
		"cg11605750",
		"cg03109729"]

group_4 = ["cg09279736",
		"cg00419321",
		"cg04663285",
		"cg13401893",
		"cg12633154",
		"cg02091185"]

# group_5 = ["cg22006386","cg06493994"]

group_5 = ["cg05412957"]

horvath_sig = ["cg14424579","cg16241714","cg02479575"]
non_sig = ["cg00009523","cg00009553","cg00009750"]
horvath_sig_2 = ["cg09809672","cg19761273","cg25809905"]
my_monsters = ["cg00000714","cg00001261","cg00004073","cg27661711","cg27664390","cg27665659","cg27665829","cg27665489"]
test = ["cg00009523"]
switch_point_like = ['cg00008387',
'cg00021786',
'cg00027400',
'cg00040423',
'cg00041872',
'cg00047469',
'cg00064235',
'cg00082384',
'cg00118317',
'cg00189220',
'cg00202441',
'cg00206356',
'cg00209398',
'cg00292435',
'cg00328219',
'cg00346392',
'cg00359395',
'cg00383384',
'cg00386725',
'cg00453202',
'cg00464927',
'cg00467553']
multipeak = ['cg00147627',
'cg00243527',
'cg00267207',
'cg00308021',
'cg00328916',
'cg00332305',
'cg00366190',
'cg00456685']
another_monsters = ['cg00101154',
'cg00158530',
'cg00236261',
'cg00277334',
'cg00374016',
'cg00443946']
male_female_diff = ['cg00029931',
'cg00040923',
'cg00050873',
'cg00126698',
'cg00139317',
'cg00170369',
'cg00205905',
'cg00258480',
'cg00347850',
'cg00363318',
'cg00379799',
'cg00390049',
'cg00403724',
'cg00409480',
'cg00419390',
'cg00426668',
'cg00469015']

test2 = ["cg00147627"]

gsm_cutoff_cgs = ["cg00113623","cg00072288","cg00004073","cg00009523"]


def load_group(list,path = "../data/Healthy_REFBASE.parquet", info_path = "../data/HEALTHY_INFO.parquet"):
    df = pq.read_table(path).to_pandas()
    df = df.loc[list]
    meth_table_info_df = pq.read_table(info_path).to_pandas().T.sort_index(axis='index')
    meth_table_df = df.T.sort_index(axis='index')
    meth_table_df_both = meth_table_df.merge(meth_table_info_df.T, left_index=True, right_index=True)
    return  meth_table_df_both

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

#df = load_new(switch_point_like)

def load_gse(list=[],gse_info_path="../data/healthy_gsm_gse_info.csv",path = "../data/Healthy_REFBASE.parquet",info_path ='../data/HEALTHY_INFO.parquet'):
    if(bool(list)):
        df = pq.read_table(path).to_pandas().loc[list]
    else:
        df = pq.read_table(path).to_pandas()
    meth_table_info_df = pq.read_table(info_path).to_pandas().T.sort_index(axis='index')
    meth_table_df = df.T.sort_index(axis='index')
    meth_table_df_both = meth_table_df.merge(meth_table_info_df.T, left_index=True, right_index=True)

    meth_table_info_df = pd.read_csv(gse_info_path).T.sort_index(axis='index')
    meth_table_info_df.columns = meth_table_info_df.loc['gsms']
    meth_table_df_both_gse = meth_table_df_both.merge(meth_table_info_df.T, left_index=True, right_index=True)

    return meth_table_df_both_gse

def load(path = "../data/Healthy_REFBASE.parquet",info_path ='../data/HEALTHY_INFO.parquet'):
    df = pq.read_table(path).to_pandas().iloc[0:10000:10]
    meth_table_info_df = pq.read_table(info_path).to_pandas()
    meth_table_df_both = df.merge(meth_table_info_df.T, left_index=True, right_index=True)
    return meth_table_df_both.T

def sep_by(df,column):
    values = list(df[column].unique())
    dfs = []
    for value in values:
        mask = df[column] == value
        filtered_df = df[mask]
        dfs.append(filtered_df)
    d = dict(zip(values, dfs))
    return d

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

def graph_cg_to_beta_dist(df, batch):
    for i in range(0, df.shape[0] - 2):
        cg = df.index[i]
        ages = df.loc['age']
        genders = df.loc['gender']
        new_df = pd.DataFrame({cg: df.iloc[i], 'age': ages, 'gender': genders})
        fig = sns.catplot(data=new_df, x="age", y=cg, hue="gender", jitter=False).fig
        fig.savefig(f'{cg}_{batch}.png')
        plt.clf()
        plt.close(fig)

def graph_cg_to_beta_dist_gse(df, batch):
    for i in range(0, df.shape[0] - 2):
        cg = df.index[i]
        ages = df.loc['age']
        gses = df.loc['gse']
        new_df = pd.DataFrame({cg: df.iloc[i], 'age': ages, 'gse': gses})
        sns.set_palette("Spectral", as_cmap=True)
        fig = sns.catplot(data=new_df, x="age", y=cg, hue="gse", jitter=False).fig
        fig.savefig(f'{cg}_{batch}.png')
        plt.clf()
        plt.close(fig)

def Extract(lst,i):
    return [item[i] for item in lst]

def arrange_stats(df):
    cgs, mean_means,mean_stds, std_means, std_stds = [],[],[],[],[]

    cgs.extend(df['cg'])
    mean_means.extend(Extract(df['mean'], 0))
    mean_stds.extend(Extract(df['mean'], 1))
    std_means.extend(Extract(df['std'], 0))
    std_stds.extend(Extract(df['std'], 1))

    new_pd = pd.DataFrame({'cg': cgs,
                           'mean_mean': mean_means,
                           'mean_std': mean_stds,
                           'std_mean': std_means,
                           'std_std': std_stds})
    new_pd.index = cgs
    return new_pd

def barplot_range_count(df,group,cgs):
    # cgs = df.index.to_list()[:len(df) - 4]
    for cg in cgs:
        working_df = df.loc[cg]
        new_pd = pd.DataFrame({
            'range': pd.cut(working_df,np.arange(0, 1, 0.01)),
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

def range_count(df,group,cgs = []):
    if bool(cgs):
        cgs = df.index.to_list()[:len(df) - 4]
    ranges, values = [], []
    for cg in cgs:
        working_df = df.loc[cg]
        new_pd = pd.DataFrame({
            'range': pd.cut(working_df,np.arange(0, 1, 0.01)),
            'val': working_df,
            'index': working_df.index,
            'counter': [1] * len(working_df)
        })

        range_count = new_pd.groupby('range')['counter'].count()
        ranges = range_count.index
        values.append(range_count.values.tolist())

    d = dict(zip(cgs,values))
    new_df = pd.DataFrame.from_dict(d)
    new_df.index = ranges
    return new_df

def peak_detection(df,prominence=8, width=3):
    for cg in df.columns:
        x = np.array(df[cg].values)
        peaks, properties = find_peaks(x, prominence=prominence, width=width)
        plt.plot(x)
        plt.plot(peaks, x[peaks], "x")
        plt.plot(np.zeros_like(x), "--", color="gray")
        plt.savefig(f"{cg}_{prominence}_{width}_peak_detection.png")
        plt.clf()

def peak_detection_2(df,threashold):
    peaks = []
    cgs = df.columns
    for cg in df.columns:
        x = df[cg].sort_index()
        top_3 = x.sort_values()[len(x) - 3:]
        top_3 = top_3.sort_index()
        peak_num = 1
        for (i,interval) in zip(range(0,2),top_3):
            lower_i = top_3.index[i].left
            lower_i2 = top_3.index[i+1].left
            if (lower_i2-lower_i > threashold):
                peak_num = peak_num + 1
        peaks.append(peak_num)
    return pd.DataFrame({'cg':cgs,'peak_num':peaks})

def peak_detection_3(df):
    for cg in df.columns:
        x = np.array(df[cg].sort_index().values)
        peakind = signal.find_peaks_cwt(data, np.arange(0, 1))
        peakind, xs[peakind], data[peakind]
        plt.plot(x)
        plt.plot(peaks, x[peaks], "x")
        plt.plot(np.zeros_like(x), "--", color="gray")
        plt.savefig(f"{cg}_{prominence}_{width}_peak_detection.png")
        plt.clf()

## filter by std and graph distributions
def filt_by_std():
    df = load()
    df_stats = calc_stats(df)
    test = arrange_stats(mean_std_total(df_stats))
    del df_stats
    test = test.sort_values(by='std_mean')
    filter_df = test.iloc[len(test)-100:len(test)]
    filter_cg = filter_df.index.to_list()
    #filter_cg = list(test[test['std_mean'] > 0.015]['cg'])
    filter_cg.extend(['age','gender'])
    filter_df = df.loc[filter_cg]
    del df
    graph_cg_to_beta_dist(filter_df,"filtered_by_std")

## density function calculation and peak detection
def dis_and_peak(df):
    range_count_df = range_count(df,"monsters")
    #peak_detection(range_count_df,prominence=20, width=0.5)
    #peak_results = peak_detection_2(range_count_df,0.02)
    peak_detection_3(range_count_df)

def gse_graph_dis():
    lists = [male_female_diff, another_monsters, multipeak, switch_point_like, my_monsters]
    lists_names = ["male_female_diff", "another_monsters", "multipeak", "switch_point_like", "my_monsters"]
    for (l,name) in zip(lists,lists_names):
        df = load_gse(l)
        #df = load_group(test2)
        #graph_cg_to_beta_dist_gse(df.T,f'{l=}'.split('=')[0])
        graph_cg_to_beta_dist_gse(df.T,name)

# distrubtion for each GSE
def dis_for_GSE(test):
    from collections import defaultdict
    df = load_gse(test)
    d = sep_by(df,"gse")
    ranges_df = defaultdict()
    for key,value in d.items():
        ranges_df[key] = range_count(value.T,key)
        #barplot_range_count(value.T,key)
#dis_for_GSE(switch_point_like)

def dis_for_GSE2():
    info_path = "../data/HEALTHY_INFO.parquet"
    gse_info_path = "../data/healthy_gsm_gse_info.csv"

    info_df = pq.read_table(info_path).to_pandas().sort_index(axis='index')
    gse_df = pd.read_csv(gse_info_path)
    gse_df.index = gse_df['gsms']
    df_merged = info_df.merge(gse_df, left_index=True, right_index=True)
    grouped = df_merged.groupby('gse')
    for name,g in grouped:
        ages_count = g.groupby('age').count()['gse']
        ax = ages_count.plot.bar(rot=90, figsize=(15, 10))
        plt.title(name)
        fig = ax.figure
        fig.savefig(f"{name}_age_count.png")
        plt.clf()
        plt.close(fig)


# gse = "GSE111629"
# gse = "GSE87571"
#gse = "GSE42861"
# gse = "GSE40239"
# df = load_new()
# df = df[df['GSE'] == gse]
# for group in [group_1,group_2,group_3,group_4,group_5]:
#     barplot_range_count(df.T,f"gsn_cutoff_cgs_{gse}",group)

# barplot_range_count(df.T,f"gsn_cutoff_cgs_{gse}",group_5)

# range_count(df,)

def read_cgs(file_path):
    with open(file_path, "r") as file:
        lines = [line.strip() for line in file]
    return lines



Tk().withdraw()
cohort_path = askopenfilename(title="Select cohort data")
cg_list = askopenfilename(title="Select cohort data")



"""
gse="GSE87571"
path=f"new_gse_test/{gse}_normalized_celltype.csv"
df = pd.read_csv(path)
df.index = df['Unnamed: 0']
df = df.drop(columns=['Unnamed: 0'])
file_path=f"AFTER_NORM/{gse}_lilifores.txt"
cgs = read_cgs(file_path)
barplot_range_count(df,gse,cgs)
"""

file_path=f"../only_health_blood_lilifores_gse111629_no_snp_cgs.txt"
#file_path=f"top10.txt"
#gse="GSE87571"
gse="GSE111629"

cgs = read_cgs(file_path)
#cgs = ['cg07148318','cg05731801','cg18105134']
#IGF2BP1
#cgs = ['cg22947322']
# GFPT2
#cgs = ['Cg01758122 ']

file_path = f"../new_gse_test/{gse}_normalized_celltype.csv"
df = pd.read_csv(file_path)
df.index = df['Unnamed: 0']
df = df.drop(columns=['Unnamed: 0'])
fit_cgs = list(set(df.index).intersection(cgs))
new_df = df.loc[fit_cgs]
barplot_range_count(new_df,gse,fit_cgs)
dis_and_peak(new_df)
