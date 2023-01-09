import numpy as np
import pandas as pd
from lxml import etree as ET
import multiprocessing as mp 
from multiprocessing import  Pool

import gender_guesser.detector as gender


def add_features(df):
    df["gender_first_author"] = df.apply(lambda row : get_name_gender(row["first_author"]), axis=1)
    df["gender_last_author"] = df.apply(lambda row : get_name_gender(row["last_author"]), axis=1)

    return df 

def parallelize_dataframe(df, func, n_cores=mp.cpu_count()):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def get_name_gender(name):

    d = gender.Detector()
    tmp_name = str(name).split(" ")
    tmp_x = [name for name in tmp_name if name!=""]
    try:
        last_name = tmp_x[0]
    except IndexError:
        return np.NaN

    if str(tmp_x[-1]).endswith("."): # avoid "J.", "L.", non-informative
        first_name = tmp_x[-2]
    else:
        first_name = tmp_x[-1]

    gender_last_name = d.get_gender(last_name)
    gender_first_name = d.get_gender(first_name)

    # print(first_name, last_name, "\t", gender_first_name, gender_last_name)

    if gender_first_name == gender_last_name:
        return str(gender_first_name)
    elif gender_first_name == "male" and gender_last_name == "female":
        return str(gender_first_name)
    elif gender_first_name == "female" and gender_last_name == "male":
        return str(gender_first_name)
    elif gender_first_name == "nan" and gender_last_name == "nan":
        return np.NaN
    elif gender_first_name == "unknown" or gender_last_name == "unknown":
        full_name = gender_last_name + gender_first_name
        full_name = full_name.replace("unknown", "")
        return str(full_name)

        

def add_gender(input_csv, output_folder = "./data"):
    " "

    sentences_df = pd.read_csv(input_csv)
    sentences_df = sentences_df.replace(r'\n','', regex=True) 

    tmp_df = sentences_df.copy() # relevant columns: last_author, first_author
    tmp_parallel_df = parallelize_dataframe(tmp_df, add_features)

    tmp_parallel_df.to_csv(f"{output_folder}/CS_v3.csv", sep=',', encoding='utf-8', index=False)



if __name__ == "__main__":

    input_csv = "./data/CS_v2.csv"
    add_gender(input_csv)
