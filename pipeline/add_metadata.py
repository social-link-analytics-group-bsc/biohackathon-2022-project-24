import requests
from requests.exceptions import HTTPError
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep
import numpy as np 

import logging
import os
import json
import yaml
from tqdm import tqdm
import pathlib
import urllib.parse

import pandas as pd
import argparse
from lxml import etree as ET
import multiprocessing as mp 
from multiprocessing import  Pool


""" 
    Script that adds the MESH terms

"""


config_path = os.path.join(os.path.dirname(
    __file__), '../config', 'config.yaml')
config_all = yaml.safe_load(open(config_path))


def get_request(url, params):
    try:
        response = requests.get(url, params)
        # If the response was successful, no Exception will be raised
        response.raise_for_status()
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'Other error occurred: {err}')
    else:
        return response



def getAnnotations(pmcid, annotation_api_url, params):
    result = get_request(annotation_api_url, params)
    root = ET.fromstring(result.content)
    return root

def retrieveAnnotations(pmcid, annotation_api_url, params):
    params = 'query={pmcid}&resultType=core&cursorMark=*&pageSize=25&format=xml'
    root = getAnnotations(pmcid, f'{annotation_api_url}search', params)
    meta_dict = {'PMCID': pmcid}
    l = []
    for m in root.iter('meshHeading'):
            l.append(m.find('descriptorName').text)
    meta_dict['MESH'] = '|'.join(list(l))

    print(meta_dict)

    return meta_dict

def get_annotations(df):
    annotation_array = []
    for idx in df.itertuples():
        annotations = retrieveAnnotations(idx.PMCID, annotation_api, xmldir)
        annotation_array.append(annotations)
    
    return annotation_array    

def parallelize_dataframe(df, func, n_cores=mp.cpu_count()):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    annotations_array = pool.map(func, df_split)
    pool.close()
    pool.join()
    return annotations_array



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script will add more data to the dataframe')
    parser.add_argument("-f", "--file", nargs=1, required=True, help="Input csv", metavar="PATH", default="./data/CS_gender_IF.csv")
    parser.add_argument("-d", "--xmldir", nargs=1, required=True, help="Directory in which xml files are stored", metavar="PATH", default="/gpfs/projects/bsc08/shared_projects/BioHackathon2022/articles_human/")
    
    annotation_api = config_all['api_europepmc_params']['rest_articles']['root_url']

    args = parser.parse_args()
    file = args.file[0]
    xmldir = args.xmldir[0]

    # df = pd.read_csv(file)

    # annotation_array = []
    # for idx in df.itertuples():
    #     annotations = retrieveAnnotations(idx.PMCID, annotation_api, xmldir)
    #     annotation_array.append(annotations)
    
    # df_ann = pd.DataFrame(annotation_array)
    # print(df_ann)

    if not os.path.isfile("./data/PMCID_MESH.csv"):
        df = pd.read_csv(file)
        annots = parallelize_dataframe(df, get_annotations) # array of dicts
        df_ann = pd.DataFrame(annots)
        df_ann.to_csv("./data/PMCID_MESH.csv") # first download the data, then manipulate it to obtain the correct format
    else:
        df_ann = pd.read_csv("./data/PMCID_MESH.csv")
        df_ann_melted = df_ann.melt("0", value_name="value").drop("value", axis=1).drop("variable", axis=1)
        df_ann_array = [json.loads(x.replace("\'", "\"")) for x in df_ann_melted.iloc[:,0].values]
        df_ann_ok = pd.DataFrame(df_ann_array)
        print(df_ann_ok.head())


    # print(df_ann.head())

    # result = pd.merge(df.reset_index(drop=True), df_ann.reset_index(drop=True), on="PMCID", how="left")
    # result.to_csv("CS_gender_IF_MESH.csv", index=False)




# def parse_xml(xmlString,id):
#     document = ET.fromstring(xmlString)
#     article_dict = {}
#     for elementtag in document.getiterator():
#         if elementtag.tag in ["year","aff","journal-id"]:
#             article_dict[elementtag.tag] = elementtag.text

#     return article_dict

# '''
# def getMetdata(pmcid, search_api_url, payload):
#     pmcid = f'PMC:{pmcid[3:]}'
#     params = urllib.parse.urlencode(payload, quote_via=urllib.parse.quote)
#     result = get_request(search_api_url, params)
#     result_json = result.json()
# '''