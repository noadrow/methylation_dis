import pyarrow.parquet as pq
import pandas as pd
import matplotlib.pyplot as plt
import os

def read_as_bulk(path = "../data/Healthy_REFBASE.parquet", batch_size=1000):
    df = pq.read_table(path).to_pandas()

    for i in range(0, len(df.columns), batch_size):
        batch = df.columns[i:i + batch_size]
        sub_df = df[batch]
        add_info_age(i,sub_df)

def add_info_age(batch_index,df,info_path ='../data/HEALTHY_INFO.parquet'):
    meth_table_info_df = pq.read_table(info_path).to_pandas().T.sort_index(axis='index')
    meth_table_df = df.T.sort_index(axis='index')
    meth_table_df_both = meth_table_df.join(meth_table_info_df.T)
    cg_list = meth_table_df.columns
    sep_by_age(batch_index,meth_table_df_both)

def sep_by_gender(batch_index,df):
    column = 'gender'
    values = list(df['gender'].unique())
    for value in values:
        mask = df[column] == value
        filtered_df = df[mask]
        filtered_df.drop(filtered_df.columns[len(filtered_df.columns) - 2], axis=1, inplace=True)
        sep_by_age(batch_index, value,filtered_df)

def sep_by_age(batch_index,gender,df):
    column = 'age'
    values = list(df['age'].unique())
    for value in values:
        mask = df[column] == value
        filtered_df = df[mask]
        filtered_df.drop(filtered_df.columns[len(filtered_df.columns) - 2], axis=1, inplace=True)
        graph_cg_to_beta_dist(batch_index,filtered_df,value,gender)

def graph_cg_to_beta_dist(batch_index,df,age,gender,chunk_size = 10):
    n_chunks = len(df)

    for i in range(n_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        sub_df = df.iloc[:,start:end]

        fig, ax = plt.subplots()
        for index, row in sub_df.iterrows():
            plt.scatter(row.index, row.values)

        plt.xticks(rotation=45, ha='right')
        plt.xlabel("Cg_id")
        plt.ylabel("Beta_Value")
        ax.grid(True)
        plt.title(f"age: {age}, gender:{gender} sample_batch: {batch_index}, cg_batch: {i}")
        #plt.show()
        if not os.path.exists(f"results/{gender}"):
            os.mkdir(f"results/{gender}")
        if not os.path.exists(f"results/{gender}/{age}"):
            os.mkdir(f"results/{gender}/{age}")
        if not os.path.exists(f"results/{gender}/{age}/{batch_index}"):
            os.mkdir(f"results/{gender}/{age}/{batch_index}")

        plt.savefig(f"results/{gender}/{age}/{batch_index}/Cgbatch_{i}.png")


horvath_sig = ["cg14424579","cg16241714","cg02479575"]
non_sig = ["cg00009523","cg00009553","cg00009750"]
read_as_bulk()