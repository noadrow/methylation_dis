import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import time
import os
def read_cgs(file_path):
    with open(file_path, "r") as file:
        lines = [line.strip() for line in file]
    return lines

def better_gap(cgs,df):
    for cg in cgs:
        range_count = range_count(cg,df)
        range_count.sort_values()[:3]

def Extract(lst,i):
    return [item[i] for item in lst]

def gauss_mode_graph(X_l,cg,save_path):

    X = np.array([[l] for l in X_l])
    # fit models with 1-3 components
    N = np.arange(1, 4)
    models = [None for i in range(len(N))]

    for i in range(len(N)):
        models[i] = GaussianMixture(N[i]).fit(X)

    # compute the AIC and the BIC
    AIC = [m.aic(X) for m in models]
    BIC = [m.bic(X) for m in models]

    fig, ax1 = plt.subplots()

    M_best = models[np.argmin(AIC)]

    x = np.arange(0, 1, 0.01)
    logprob = M_best.score_samples(x.reshape(-1, 1))
    responsibilities = M_best.predict_proba(x.reshape(-1, 1))
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    ax1.hist(X, 30, density=True, histtype='stepfilled', alpha=0.4)
    ax2 = ax1.twinx()
    ax2.plot(x, pdf_individual, '--k')

    #plt.show()


    plt.savefig(os.path.join(save_path,f"{cg}.png"))

    '''
    M_best = models[np.argmin(AIC)]
    x = np.linspace(0, 1, 1000)
    logprob = M_best.score_samples(x.reshape(-1, 1))
    responsibilities = M_best.predict_proba(x.reshape(-1, 1))
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    if (len(pdf_individual[0])>2):
        X0, X1, X2 = Extract(pdf_individual, 0), Extract(pdf_individual, 1), Extract(pdf_individual, 2)
    elif(len(pdf_individual[0])>1):
        X0, X1, X2 = Extract(pdf_individual, 0), Extract(pdf_individual, 1), len(pdf_individual)*[None]
    else:
        X0, X1, X2 = Extract(pdf_individual, 0),len(pdf_individual)*[None], len(pdf_individual)*[None]

    res = stats_out([X0,X1,X2])
    return [X,pdf_individual]
    '''


def stats_out(pdf_list):
    res = []
    for l in pdf_list:
        if l[0] == None:
            res.append([None,None,None,None])
        else:
            res.append([np.array(l).mean(), np.array(l).std(), np.array(l).max(), np.array(l).sum()])

    return res

def gup_hunt(cgs,df):
    new_df = pd.DataFrame()
    for cg in cgs:
        new_df[cg] = df.loc[cg].sort_values().diff()

    mask = new_df > 0.16
    mask = mask.sum()
    mask_3 = mask[mask > 1].index.to_list()
    mask_2 = mask[mask == 1].index.to_list()

    return [mask_2, mask_3]

def range_count(df,cg):
    working_df = df.loc[cg]
    new_pd = pd.DataFrame({
        'range': pd.cut(working_df, np.arange(0, 1, 0.01)),
        'val': working_df,
        'index': working_df.index,
        'counter': [1] * len(working_df)
    })

    range_count = new_pd.groupby('range')['counter'].count()
    return range_count

def barplot_range_count(df,group,cgs):
    # cgs = df.index.to_li
    # st()[:len(df) - 4]
    for cg in cgs:
        range_count = range_count(df,cg)

        ax = range_count.plot.bar(rot=90,figsize=(15,10))
        fig = ax.figure
        fig.savefig(f"{cg}_{group}_barh.png")
        plt.clf()
        plt.close(fig)

def load_pickle_to_df(path):
    from tkinter.filedialog import askopenfilename
    import pickle
    import time

    print("Waiting for loading the file")
    file = open(path, 'rb')
    print("File load")
    print("Waiting for pickle to convert to dataframe")
    t0 = time.time()
    df = pickle.load(file)
    df.index = df.iloc[:,0]
    df = df.drop(df.columns[0], axis=1)
    print(f"Dataframe loaded: {time.time()-t0} sec")
    file.close()
    return df

def to_dict(**kwargs):
    d = {k:v for k,v in kwargs.items() if v is not None}
    return d

def output_log(d,OUTPUT):
    import time
    import os
    from datetime import datetime
    import json

    now = str(datetime.now()).replace(" ","_").replace(":","-")
    folder_pos = os.path.dirname(OUTPUT)
    log_path = os.path.join(folder_pos,f'log_{now}.json')
    with open(log_path, 'w') as json_file:
        json.dump(d, json_file)
    print(f"log file were saved at {log_path}")

def load_input_from_user():
    print("waiting for data")
    print("...")
    data_path = askopenfilename(title="Choose a pickle")
    print("waiting for CpG list")
    print("...")
    cg_path = askopenfilename(title='Insert txt list (CpG list)')
    print("waiting for saving folder")
    print("...")
    save_path = filedialog.askdirectory(title="Select Folder for saving the graphs")

    print("loading data")
    print("...")
    df = load_pickle_to_df(data_path)
    print("loading cgs")
    print("...")
    cgs = read_cgs(cg_path)

    output_log(to_dict(cg_path=cg_path,save_path=save_path,cgs=cgs,data_path=data_path),OUTPUT=save_path)
    return cgs,df,save_path

def load_input_from_json(json_path):
    import json
    with open(json_path, 'r') as json_file:
        conf_dict = json.load(json_file)
    return [conf_dict['cg_path'],conf_dict['save_path'],conf_dict['cgs'],conf_dict['data_path']]

print("waiting for log (re-run pipeline)")
print("...")
json_path = askopenfilename(title='Insert json log file (to re-run pipline, else close the window)')

if (json_path):
    print("initiating log's configuration INPUTS")
    print("...")
    cg_path,save_path,cgs,data_path = load_input_from_json(json_path)
    df = load_pickle_to_df(data_path)

else:
    print("manual input rout")
    print("...")
    cgs,df,save_path = load_input_from_user()


print("GMM and graph outputs initiated")
print("...")

t0 = time.time()
for cg in cgs:
    if (cg in df.index):
        if(len(df.loc[cg])>0):
            #AIC,models,X = gauss_mode(range_count(df,cg))
            gauss_mode_graph(X_l=df.loc[cg],save_path=save_path,cg=cg)
    else:
        print("cg is not in the table")

timer = time.time()-t0
print(f"finished GMM process in {timer} sec")
