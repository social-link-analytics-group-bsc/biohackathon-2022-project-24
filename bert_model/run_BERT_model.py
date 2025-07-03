from transformers import pipeline
from pprint import pprint
import csv
import argparse
import json
from tqdm import tqdm
import sqlite3
import sys
import os
import yaml
import nltk
from nltk.tokenize import sent_tokenize
import concurrent.futures
import re
from word2number import w2n

def parsing_arguments(parser):
    parser.add_argument("--model", type=str, default='output/bert-base-uncased-en/sbe.py_8_0.00005_date_22-11-10_time_14-55-26',
                        help='Pretrained model to find the numbers')
    return parser
 
# Method for getting database entries that still need to be run through the model
def get_entries(conn, cur):
    # SQL_QUERY = 'SELECT CASE WHEN COUNT(*) > 0 THEN 0 ELSE 1 END as IsEmpty FROM Results;'
    # cur.execute(SQL_QUERY)
    # result = cur.fetchone()

    # # If results table is empty, start with all records
    # if result[0]:
    #     print("Results is empty, start from the beginning.")
    #     SQL_QUERY = 'SELECT sections.pmcid, sections.METHODS FROM sections;'
    #     cur.execute(SQL_QUERY)
    # else:
    #     print("Results is not empty, fetch methods still needing to be processed.")
    #     SQL_QUERY = 'SELECT sections.pmcid, sections.METHODS FROM sections LEFT JOIN status ON sections.pmcid = status.pmcid WHERE status.model_results IS NULL;'
    #     cur.execute(SQL_QUERY)
    # return cur.fetchall()

    # FOR TESTING
    # SQL_QUERY = 'SELECT sections.pmcid, sections.METHODS FROM sections LIMIT 10'
    # cur.execute(SQL_QUERY)
    # return cur.fetchall()

    # For BH2023
    pmcid_list = []

    with open('pmcid_list.txt', 'r') as file:
        for line in file:
            pmcid_list.append(str(line.strip()))
    
    # Create the table with a single column for IDs
    cur.execute(f"CREATE TABLE IF NOT EXISTS cordis_pmcid_list (pmcid TEXT NOT NULL)")
    print("Created cordis pmcid table")

    # Insert the IDs into the table
    for id_value in pmcid_list:
        cur.execute(f"INSERT INTO cordis_pmcid_list (pmcid) VALUES (?)", (id_value,))
    print("Added all pmcids to cordis table")

    # Commit the changes to the database
    conn.commit()

    SQL_QUERY = f"SELECT sections.pmcid, sections.METHODS FROM sections INNER JOIN cordis_pmcid_list ON sections.pmcid = cordis_pmcid_list.pmcid;"
    result = cur.execute(SQL_QUERY)
    return cur.fetchall()

# Create table to record model results in (pmcid, n_fem, n_male, per_fem, perc_male, sample)
def create_results_table(conn, cur):
    cur.execute(
        """CREATE TABLE IF NOT EXISTS "model_output" (
                "pmcid"	TEXT NOT NULL,
                "sentence_index"	INTEGER,
                "n_fem"	TEXT,
                "n_male"	TEXT,
                "perc_fem"	TEXT,
                "perc_male"	TEXT,
                "sample"	TEXT,
                FOREIGN KEY ("pmcid") REFERENCES sections("pmcid")
            )"""
        )
    conn.commit()

# Add new row to results  
def add_result(conn, cur, pmcid, results):

    SQL_QUERY = "INSERT INTO model_output (pmcid, sentence_index, n_male, n_fem, perc_male, perc_fem, sample) VALUES  (?, ?, ?, ?, ?, ?, ?)"
    cur.execute(SQL_QUERY, tuple(results))
    update_status(conn, cur, pmcid)

def update_status(conn, cur, pmcid):
    SQL_QUERY = "UPDATE status SET model_results = 1 WHERE pmcid = ?;"
    cur.execute(SQL_QUERY, (pmcid,))
    conn.commit()

def process_list(input_list):
    result_list = []
    for item in input_list:
        try:
            number = int(item)
            result_list.append(number)
        except ValueError:
            # If it is not a valid integer, try a float
            try:
                number = float(item)
                result_list.append(number)
            except ValueError:
                # If it's not an integer or float, try word2number
                try:
                    number = w2n.word_to_num(item)
                    result_list.append(number)
                except ValueError:
                    # If word2number also fails, discard the item
                    pass
    
    return result_list

def run_model_on_entry(row, nlp):
    pmcid, methods = row
    results = []
    # Split methods section into sentences
    if methods is not None:
        sentences = sent_tokenize(methods)
        modified_sentences = []
        for sentence in sentences:
            if len(sentence) > 512:
                sentence = sentence[:512]
            modified_sentences.append(sentence)

        annotations = nlp(modified_sentences)
        
        sentence_index = []
        n_male = []
        n_fem = []
        perc_male = []
        perc_fem = []
        sample = []

        sent_index = 0

        for list in annotations:
            if list:
                dict = {'pmcid' : pmcid, 'sentence_index' : sent_index, 'n_male' : [], 'n_fem' : [], 'perc_male' : [], 'perc_fem' : [], 'sample' : []}
                list_index = 0
                for annotation in list:
                    # if the token we are looking at should be appended to the previous token
                    if (list_index > 0) and (annotation["entity"] == list[list_index-1]["entity"]) and (annotation['word'].startswith("##")) and (list[list_index-1]["end"] == annotation["start"]):
                        # Get the last entry added to the list for this entity in the dictionary (this is what we need to append to)
                        prev = dict[annotation["entity"]][-1]
                        # Remove hashtags
                        add = annotation["word"].replace("#", "")
                        # Append current number to previous numbers
                        dict[annotation["entity"]][-1] = prev + add
                    else:
                        dict[annotation["entity"]].append(annotation["word"])
                    list_index += 1

                #TO DO: clean values before adding? Clean words into ints and turn list of strings into list of integers
                sentence_index.append(sent_index)
                n_male.append(process_list(dict["n_male"]))
                n_fem.append(process_list(dict["n_fem"]))
                perc_male.append(process_list(dict["perc_male"]))
                perc_fem.append(process_list(dict["perc_fem"]))
                sample.append(process_list(dict["sample"]))
            sent_index += 1

        results = [pmcid, json.dumps(sentence_index), json.dumps(n_male), json.dumps(n_fem), json.dumps(perc_male), json.dumps(perc_fem), json.dumps(sample)]

    return pmcid, results

def main():
    parser = argparse.ArgumentParser()
    parser = parsing_arguments(parser)
    args = parser.parse_args()
    print(args)
    print('Loading the data...')

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(os.path.dirname(__file__), "../config", "config.yaml")
    config_all = yaml.safe_load(open(config_path))

    # Connect to the SQLite database
    print('Connecting to DB...')
    DB_FILE = config_all["api_europepmc_params"]["db_info_articles"]
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    # Create results table (if not already created)
    create_results_table(conn, cur)
    print("Created new results table")

    # Get entries from db that need to be run through the model
    print("Getting entries...")
    entries = get_entries(conn, cur)
    print("Got entries to be processed: ", len(entries))

    # Run entries through the model (sentence by sentence? check this)
    nlp = pipeline("ner", model=args.model, device=0)
    #nlp = pipeline("ner", model=args.model) # if you are working locally, remove device=0

    print("Running the model on entries with multiple threads...")
    executor = concurrent.futures.ThreadPoolExecutor()
    futures = [
        executor.submit(run_model_on_entry, row, nlp) for row in entries
    ]

    pbar = tqdm(total=len(entries))
    for future in concurrent.futures.as_completed(futures):
        pmcid, results = future.result()
        if results:
            add_result(conn, cur, pmcid, results)
        else:
            # If no results found by model but methods were processed, just update the status
            update_status(conn, cur, pmcid)
        pbar.update(n=1)

    conn.close()


if __name__ == "__main__":
    main()