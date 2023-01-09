
import os
import numpy as np
import pandas as pd
from lxml import etree as ET
import multiprocessing as mp 
from multiprocessing import  Pool
from impact_factor import ImpactFactor

def add_features(df):

    df["impact_factor"] = df.apply(lambda row : get_impact_factor(row["journal-title"]), axis=1)
    # df["impact_factor_mean_2015"] = df.apply(lambda row : get_impact_factor(row["journal-title"])[1], axis=1)
    
    return df 

def parallelize_dataframe(df, func, n_cores=mp.cpu_count()):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def get_impact_factor(journal_name):
    " adds IF of journal to the provided journal title"

    IF = ImpactFactor()
    journal_name = str(journal_name).strip()
    if_dict = IF.search(journal_name)
    # print(journal_name, if_dict)

    if if_dict is not None:
        impact = if_dict["factor"]
        mean_impact_2015_list = if_dict["factor_history"].items()
        mean_impact_2015 = np.array([x[1] for x in mean_impact_2015_list]).mean()

        # print(journal_name, impact)

        return impact
    else:
        return np.NaN
        

def add_IF(input_csv, output_folder = f"{os.getcwd()}/data"):
    " "

    sentences_df = pd.read_csv(input_csv)
    sentences_df = sentences_df.replace(r'\n','', regex=True) 

    tmp_df = sentences_df.copy() # relevant columns: last_author, first_author
    tmp_parallel_df = parallelize_dataframe(tmp_df, add_features)

    tmp_parallel_df.to_csv(f"{output_folder}/CS_v2.csv", sep=',', encoding='utf-8', index=False)




if __name__ == "__main__":


    input_csv = f"{os.getcwd()}/CS_v1.csv"
    add_IF(input_csv)

