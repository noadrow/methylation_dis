import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load(gse):
    #  load data
    file_path=f"AFTER_NORM/only_multipeak_cgs_{gse}.csv"
    table_1 = pd.read_csv(file_path)
    file_path=f"AFTER_NORM/infinium.bed"
    table_2 = pd.read_csv(file_path,sep="\t")

    # arrange data
    df = pd.merge(table_1, table_2, left_on='cg', right_on='Name', how='outer')
    df = df.dropna()
    df.drop('Name', axis=1, inplace=True)

    df['chr_num'] = df.CHR_hg38.str.extract('(\d+)')
    df['chr_num'] = df.apply(lambda row: row['CHR_hg38'] if pd.isna(row['chr_num']) or row['chr_num'] == '' else row['chr_num'], axis=1)
    # df['chr_num'] = df.apply(lambda row: 23 if pd.isna(row['chr_num']) or row['chr_num'] == '' else row['chr_num'], axis=1)
    df['chr_num'] = df['chr_num'].apply(lambda x: '0' + x if len(x) < 2 else x)
    df = df.sort_values('chr_num')

    return df

# manhattan plot
def plot_manhatten(type,gse):
    df['ind'] = range(len(df))
    df_grouped = df.groupby('chr_num')
    fig = plt.figure(figsize=(14, 8)) # Set the figure size
    ax = fig.add_subplot(111)
    colors = ['black','grey','black', 'grey']
    x_labels = []
    x_labels_pos = []
    for num, (name, group) in enumerate(df_grouped):
        group.plot(kind='scatter', x='ind', y=type,color=colors[num % len(colors)], ax=ax)
        last = (group['ind'].iloc[-1] - (group['ind'].iloc[-1] - group['ind'].iloc[0])/2)
        x_labels.append(name)
        if name == "chrX":
            x_labels_pos.append(last)
    ax.set_xticks(x_labels_pos)
    ax.set_xticklabels(x_labels)

    # set axis limits
    ax.set_xlim([0, len(df)])
    ax.set_ylim([0, 3.5])

    # x axis label
    ax.set_xlabel('Chromosome')

    plt.title(gse)
    # show the graph
    plt.savefig(f"{type}_{gse}.png")

def count_on_chr(gse):
    df['ind'] = range(len(df))
    df_grouped = df.groupby('chr_num')['chr_num'].count()
    df_grouped.plot(kind='bar')
    plt.title(f'{gse}: number of passed sites on each chromosome')
    plt.savefig(f'{gse}_count_chromosome.png')



#gse="GSE87571"
gse="GSE111629"
df = load(gse)
# count_on_chr(gse)

type = 'log10_lilli'
df[type] = -np.log10(df['satistics'])
df = df.sort_values(by=type)
df = df[df['chr_num'] != 'chrX']
df.iloc[:10]['cg'].to_csv(f'{gse}_bottom10.txt',index=False)
df.iloc[-10:]['cg'].to_csv(f'{gse}_top10.txt',index=False)
"""
gse = f"{gse}_with_threashold"

type = 'lilifores_satistics'
df[type] = df['satistics']
plot_manhatten(type,gse)

type = 'standart_deviation'
df[type] = df['std']
plot_manhatten(type,gse)
"""

