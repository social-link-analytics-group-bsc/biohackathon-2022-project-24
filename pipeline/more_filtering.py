
from enum import unique
import logging
import os
import json
import yaml
from tqdm import tqdm
import pathlib
import urllib.parse
import numpy as np

import pandas as pd
import argparse
from lxml import etree as ET
import multiprocessing as mp

""" script to add more info """


config_path = os.path.join(os.path.dirname(
    __file__), '../config', 'config.yaml')
config_all = yaml.safe_load(open(config_path))


def parse_xml(xmlString,id):
    document = ET.fromstring(xmlString)
    article_dict = {}

    elementtag_list = [] # debugging purposes
    names_list = []
    subject_list = []
    kwds_list = []

    name_count = 0

    for elementtag in document.getiterator():
        elementtag_list.append(elementtag.tag)

        unique_elements = set(elementtag_list)
        # print(unique_elements)
 
        # looking at the citation: ref-count, tr, corresp
        # if elementtag.tag in ["article-meta"]:
        #     institution_list = [str(x.text).strip().replace("\n", " ").replace(",", "") for x in elementtag.iter()]
        #     institut = institution_list

        #     print(institut)

            # article_dict["institution"] = institut 





        # finding the type of article

        for body in document.findall('.//body/'):
            article_type = str(body.attrib)
            article_type = article_type.replace("\'", "\"")
        

        # TODO: this step can be encapsulated in less code, surely 
        article_type_list = json.loads(article_type)
        article_type_list = list(article_type_list.items())
        type_of_article = article_type_list[0][1]

        article_dict["article_type"] = type_of_article

        if elementtag.tag in ["institution"]:
            institution_list = [str(x.text).strip().replace("\n", " ").replace(",", "") for x in elementtag.iter()]
            institut = institution_list[0]

            article_dict["institution"] = institut 
        
        
        if elementtag.tag in ["subj-group"]:
            subj_list = [str(x.text).strip().replace("\n", " ") for x in elementtag.iter()]
            subj_list = list(filter(None, subj_list))
            subj_string = ", ".join(subj_list)

            article_dict["subject"] = subj_string
        

        if elementtag.tag in ["funding-group"]:
            tmp_array = [str(x.text).strip().replace("\n", " ") for x in elementtag.iter()]
            tmp_array = list(filter(None, tmp_array))
            funding_string = ", ".join(tmp_array)

            article_dict["funding"] = funding_string
        
        if elementtag.tag in ["country"]:
            tmp_array = [str(x.text).strip().replace("\n", " ") for x in elementtag.iter()]
            tmp_array = list(filter(None, tmp_array))
            string = " ,".join(tmp_array)

            article_dict["country"] = string
        
        if elementtag.tag in ["award-id"]:
            tmp_array = [str(x.text).strip().replace("\n", " ") for x in elementtag.iter()]
            tmp_array = list(filter(None, tmp_array))
            string = " ,".join(tmp_array)

            article_dict["award"] = string

        if elementtag.tag in ["issn"]:
            tmp_array = [str(x.text).strip().replace("\n", " ") for x in elementtag.iter()]
            tmp_array = list(filter(None, tmp_array))
            string = " ,".join(tmp_array)

            article_dict["ISSN"] = string
        
        

        # parsing authors
        if elementtag.tag in ["contrib"]:
            names_list = [x.text for x in elementtag.iter() if x is not None]
            # print(names_list)
            if name_count == 0:
                try:
                    name = (" ").join(names_list[2:4]).strip().replace("\n", " ")
                    article_dict["first_author"] = name
                except:
                    article_dict["first_author"] = np.NaN

            name_count += 1
            article_dict[elementtag.tag] = name_count

        try:
            name = ("").join(names_list[2:4]).replace("\n", "").strip()
            article_dict["last_author"] = name
        except:
            article_dict["last_author"] = np.NaN

        # parsing keywords
        if elementtag.tag in ["kwd-group"]: # kwd-group
            kwds_list = [x.text for x in elementtag.iter() if x.text is not None]
        try:
            article_dict["keywords"] = (" ,").join(kwds_list).strip()
        except:
            article_dict["keywords"] = kwds_list
        

        # parsing the rest of elements
        if elementtag.tag in ["year", "aff", "publisher-name", "journal-title", "article-title"]: 
            names_list.append(elementtag.text)
            author_len = len([x.text for x in elementtag.iter() if x is not None])
            article_dict[elementtag.tag] = str(elementtag.text).strip()
        
    return article_dict

def build_complete_df(df):
    tmp_list = []
    for idx in df.itertuples():
        id = idx.PMCID
        # print(id, end="\t")
        filename=str(idx.PMCID)+".xml"

        with open(directory +filename,"r") as xml:
                s_xml = xml.read()
                article_d = parse_xml(s_xml,id)        
                article_d.update(idx._asdict())
                
        article_d.pop("Index")
        # print(filename)
        tmp_list.append(article_d)
        
    df = pd.DataFrame.from_dict(tmp_list)
        
    return df

def parallelize_dataframe(df, func, n_cores=mp.cpu_count()):
    df_split = np.array_split(df, n_cores)
    pool = mp.Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()

    return df



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script will add more data to the dataframe')
    parser.add_argument("-f", "--file", nargs=1, required=True, help="Input csv", metavar="PATH", default="./data/CS_pre.csv")
    parser.add_argument("-d", "--directory", nargs=1, required=True, help="Directory in which xml files are stored", metavar="PATH", default="/gpfs/projects/bsc08/shared_projects/BioHackathon2022/articles_human/")


    args = parser.parse_args()
    file = args.file[0]
    directory = args.directory[0]
    #annotation = args.annotation[0]

    df = pd.read_csv(file)
    names = 0

    df_array = parallelize_dataframe(df, build_complete_df)
    # tmp_parallel_list = [x[0] for x in tmp_parallel_list]

    # df_array = pd.DataFrame.from_dict(tmp_parallel_list)
    df_array.replace("\n", " ", inplace = True)

    print(df_array.info())

    df_array.to_csv("./data/CS_v0.csv", sep=",", index=False)


    # if elementtag.tag in ["subject"]:
        #     try:
        #         subject_list = [str(x.text).strip() for x in elementtag.iter() if "Article" not in x.text]

        #         if len(subject_list) != 0 and "Research" in subject_list[0]:
        #             subject_list = "None"
        #         if len(subject_list) != 0 and "Letters" in subject_list[0]:
        #             subject_list = "None"

        #     except:
        #         subject_list = "None"

        #     if len(article_dict):
        #         article_dict["subjects"] = subject_list