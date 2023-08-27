import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tkinter import Tk
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter.simpledialog import askinteger

import os

Tk().withdraw()

def read_cgs(file_path):
    with open(file_path, "r") as file:
        lines = [line.strip() for line in file]
    return lines
def graph_cg_to_beta_dist(df, batch,path):
    for i in range(0, df.shape[0] - 2):
        cg = df.index[i]
        ages = df.loc['age']
        genders = df.loc['gender']
        new_df = pd.DataFrame({cg: df.iloc[i], 'age': ages, 'gender': genders})
        fig = sns.catplot(data=new_df, x="age", y=cg, hue="gender", jitter=False).fig
        fig.savefig(f'{path}/{cg}_{batch}.png')
        plt.clf()
        plt.close(fig)

def barplot_range_count(df,group,relative_path):
    cgs = df.index.to_list()
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
        fig.savefig(f"{relative_path}/{cg}_{group}_barh.png")
        plt.clf()
        plt.close(fig)

relative_path = filedialog.askdirectory(initialdir=os.getcwd(),title="choose data folder")
gse = "GSE40279"
df = pd.read_csv(f"{relative_path}/{gse}_normalized_celltype.csv",index_col=0)
print("df loaded")
df_info = pd.read_csv(f"{relative_path}/{gse}_info.csv",index_col=0)
print("info df loaded")
df_filt = df_info[df_info['tissue'] == 'whole blood'][df_info['ethnicity'] == 'Caucasian - European'].T
vals = df_filt.loc['source_name'].values.tolist()
print("info df filtered")
selected_rows = df.T[df.T.index.isin(vals)]
print("df filtered")
cg_path = askopenfilename(title="choose cg list")
cgs = read_cgs(cg_path)
print("load cgs")
df_filt = selected_rows.T[selected_rows.T.index.isin(cgs)]
print("filter df by cgs")
relative_path = filedialog.askdirectory(initialdir=os.getcwd(),title="choose where to save graphs")
#graph_cg_to_beta_dist(df_filt,gse,relative_path)
barplot_range_count(df_filt,gse,relative_path)
print("graph")
