import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter import filedialog
from tkinter.simpledialog import askstring
from tkinter.filedialog import askopenfilename
def barplot_range_count(df, group, cgs):
    # cgs = df.index.to_list()[:len(df) - 4]
    for cg in cgs:
        working_df = df.loc[cg]
        new_pd = pd.DataFrame({
            'range': pd.cut(working_df, np.arange(0, 1, 0.01)),
            'val': working_df,
            'index': working_df.index,
            'counter': [1] * len(working_df)
        })

        range_count = new_pd.groupby('range')['counter'].count()

        ax = range_count.plot.bar(rot=90, figsize=(15, 10))
        fig = ax.figure
        fig.savefig(f"{cg}_{group}_barh.png")
        plt.clf()
        plt.close(fig)

def read_cgs(file_path):
    with open(file_path, "r") as file:
        lines = [line.strip() for line in file]
    return lines

Tk().withdraw()
cg_path = askopenfilename(title="Select cg list")
save_path = filedialog.askdirectory(title="Select Folder for output")
gse = askstring(title='select gse',prompt='select gse')
path = f'C:/Users/User/Documents/blood_methylation/TEST1/data/new_gse_test/{gse}_normalized_celltype.csv'
df = pd.read_csv(path,index_col=0)
cgs = read_cgs(cg_path)
barplot_range_count(df, gse, cgs)

df_log = pd.DataFrame({'gse':gse,'cgs':cgs,'save_folder':save_path})
df_log.to_csv(f'{path}/log_{gse}_{cgs}.txt')
