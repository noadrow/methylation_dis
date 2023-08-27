import pandas as pd
from statsmodels.stats.diagnostic import lilliefors
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
def read_cgs(file_path):
    with open(file_path, "r") as file:
        lines = [line.strip() for line in file]
    return lines
# collect data from the official illumina table by selected cgs
def illumi_table():
    Tk().withdraw()
    path = filedialog.askdirectory(title="Select Root Folder")
    cg_path = askopenfilename(title="Select cg list")

    df = pd.read_csv(f"{path}/full_illumina.csv")

    filter_cgs = read_cgs(cg_path)
    df.index = df["IlmnID"]
    cgs_intersect = df.index.intersection(filter_cgs)

    after_filter = df[ df["IlmnID"].isin(cgs_intersect)]

    after_filter.to_csv(f"{path}/filter_illumina_{cg_path.split('/')[-1].split('.')[0]}.csv")

def data_prep(gse):
    #gses = ["GSE87571","GSE107737","GSE87648","GSE111629]
    file_path = f"{gse}_normalized_celltype.csv"
    df = pd.read_csv(file_path,index_col=0)

    info_path = f'{gse}_info.csv'
    meth_table_info_df = pd.read_csv(info_path).sort_index(axis='index')
    meth_table_info_df.index = meth_table_info_df['source_name']
    meth_table_df = df.T.sort_index(axis='index')
    meth_table_df_both = meth_table_df.merge(meth_table_info_df,left_index=True, right_index=True)
    return meth_table_df_both

def lilliefors_test_fast(df,gse):
    results,cgs,satistics,stds = [], [], [],[]
    for cg in df.index[:len(df)]:
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

def gup_hunt(cgs,df):
    new_df = pd.DataFrame()
    for cg in cgs:
        new_df[cg] = df.loc[cg].sort_values().diff()

    mask = new_df > 0.16
    mask = mask.sum()
    mask_3 = mask[mask > 1].index.to_list()
    mask_2 = mask[mask == 1].index.to_list()

    return [mask_2, mask_3]

illumi_table()

#gse = "GSE40279"
#gse = "GSE87571"
#df = pd.read_csv("GSE87571_normalized_celltype.csv",index_col=0)
#lilliefors_test_fast(df,gse)
#gup_hunt(df.index.to_list(),df)


"""
gse = "GSE40279"
df = pd.read_csv(f"{gse}_normalized_celltype.csv",index_col=0)
print("df loaded")
df_info = pd.read_csv(f"{gse}_info.csv",index_col=0)
print("info df loaded")
df_filt = df_info[df_info['tissue'] == 'whole blood'][df_info['ethnicity'] == 'Caucasian - European'].T
vals = df_filt.loc['source_name'].values.tolist()
print("info df filtered")
selected_rows = df.T[df.T.index.isin(vals)]
print("df filtered")
lilliefors_test_fast(selected_rows.T,gse)
print("lilifroes done")

"""


"""

cg_path = "../no_snp/try2.cpgs.txt"
cg_path = "../only_health_blood_lilifores_gse111629_no_snp_cgs.txt"
gse = "gse111629"
illumi_table(cg_path,gse)

gse = "GSE87571"
df=data_prep()
filtered_df = df[df['disease state'] == 'normal'][df['tissue'] == 'whole blood']
file_path = f"../new_gse_test/{gse}_normalized_celltype.csv"
df = pd.read_csv(file_path,index_col=0)
lilliefors_test_fast(df,f"only_blood_health_df_{gse}")




info_path = f'../new_gse_test/{gse}_info.csv'
filtered_df = info_df[info_df['disease state'] == 'normal'][info_df['tissue'] == 'whole blood']

#age_30_group = meth_table_df_both[meth_table_df_both['age'] > 29][meth_table_df_both['age'] < 36]
#age_50_group = meth_table_df_both[meth_table_df_both['age'] > 44][meth_table_df_both['age'] < 51]
#lilliefors_test_fast(age_30_group,"age_30_group")
#lilliefors_test_fast(age_50_group,"age_50_group")

# no healthy samples
# gse = "GSE87571"
filtered_df[filtered_df['gender'] == 'Female']
filtered_df[filtered_df['gender'] == 'Male']



#filtered_df_gen = filtered_df[filtered_df['gender'] == 'Female']
#plt.hist(filtered_df_gen['age'])
#plt.title(f'{gse} age distribution')
#plt.ylabel('freq')
#plt.xlabel('age')


import matplotlib.pyplot as plt

#there is healthy samples in GSE107737 only males
gse = "GSE107737"
info_path = f'../new_gse_test/{gse}_info.csv'
info_df = pd.read_csv(info_path)
filtered_df = info_df[info_df['diagnosis'] == 'Normal'][info_df['tissue'] == 'whole blood']
filtered_df_age = filtered_df['age'].str.extract('(\d+)').astype(int)
filtered_df['age'].hist()
plt.title(f'{gse} age distribution')
plt.ylabel('freq')
plt.xlabel('age')



gse = "GSE87648"
[info_df['cell type'] == 'Whole blood']
simplified_diagnosis

path = 'C:/Users/User/Documents/blood_methylation/TEST1/data/new_gse_test/GSE40279_normalized_celltype.csv'
df = pd.read_csv(path,index_col=0)

gse = "GSE40279"
path = f'C:/Users/User/Documents/blood_methylation/TEST1/data/new_gse_test/{gse}_info.csv'
df = pd.read_csv(path,index_col=0)

#df_t = df[df['disease state']=='PD-free control'][df['tissue']=='whole blood'][df['ethnicity']=='Caucasian']
#font = {'family' : 'normal',
#        'weight' : 'bold',
#        'size'   : 24}

#gse = 'GSE111629 
#matplotlib.rc('font', **font)
#df_f = df_t[df_t['gender']=='Female']
#df_m = df_t[df_t['gender']=='Male']
#df_m['age'].hist(alpha=0.5, color='red',label='male')
#df_f['age'].hist(alpha=0.5, color='blue',label='female')
#plt.title(f'{gse} age distribution')
#plt.ylabel('freq')
#plt.xlabel('age')
#plt.legend()

M_df = df[df['gender'] == 'M'][df['tissue'] == 'whole blood'][df['ethnicity'] == 'Caucasian - European']
F_df = df[df['gender'] == 'F'][df['tissue'] == 'whole blood'][df['ethnicity'] == 'Caucasian - European']
M_df['age (y)'].hist(alpha=0.5, color='red',label='male')
F_df['age (y)'].hist(alpha=0.5, color='blue',label='female')
plt.title(f'{gse} age distribution')
plt.ylabel('freq')
plt.xlabel('age')
plt.legend()

#cg_path="../only_health_blood_lilifores_gse111629_no_snp_cgs.txt"
#illumi_table(cg_path)

meth_table_df_both = data_prep()
# [205 rows x 232335 columns]
only_blood_health_df = meth_table_df_both[meth_table_df_both['tissue'] == 'whole blood'][meth_table_df_both['disease state'] == 'PD-free control']

df_m = df[df['tissue']=='whole blood'][df['disease state']=='normal'][df.index=='Male']
df_f = df[df['tissue']=='whole blood'][df['disease state']=='normal'][df.index=='Female']

"""

