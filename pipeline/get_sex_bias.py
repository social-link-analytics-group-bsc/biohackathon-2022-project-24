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


def parsing_arguments(parser):
    parser.add_argument("--model", type=str, default='output/bert-base-uncased-en/sbe.py_8_0.00005_date_22-11-10_time_14-55-26',
                        help='Pretrained model to find the numbers')
    return parser
 
# Method for getting database entries that still need to be run through the model
def get_entries(conn, cur):
    SQL_QUERY = 'SELECT CASE WHEN COUNT(*) > 0 THEN 0 ELSE 1 END as IsEmpty FROM Results;'
    cur.execute(SQL_QUERY)
    result = cur.fetchone()

    # If results table is empty, start with all records
    if result[0]:
        SQL_QUERY = 'SELECT sections.pmcid, sections.METHODS FROM sections;'
        cur.execute(SQL_QUERY)
    else:
        SQL_QUERY = 'SELECT sections.pmcid, sections.METHODS FROM sections LEFT JOIN status ON sections.pmcid = status.pmcid WHERE status.model_results IS NULL;'
        cur.execute(SQL_QUERY)
    return cur.fetchall()

# Create table to record model results in (pmcid, n_fem, n_male, per_fem, perc_male, sample)
def create_results_table(conn, cur):
    cur.execute(
        """CREATE TABLE IF NOT EXISTS "Results" (
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

    SQL_QUERY = "INSERT INTO Results (pmcid, sentence_index, n_male, n_fem, perc_male, perc_fem, sample) VALUES  (?, ?, ?, ?, ?, ?, ?)"
    for row in results:
        cur.execute(SQL_QUERY, row)
    update_status(conn, cur, pmcid)

def update_status(conn, cur, pmcid):
    SQL_QUERY = "UPDATE status SET model_results = 1 WHERE pmcid = ?;"
    query_input = (pmcid)
    cur.execute(SQL_QUERY, query_input)
    conn.commit()

def run_model_on_entry(row, nlp):
    pmcid, methods = row
    results = []
    # Split methods section into sentences
    if methods is not None:
        sentences = sent_tokenize(methods)
        # Establish array of tuples of results per sentence
        #results = []
        index = 0
        # Loop through each sentence
        for sentence in sentences:
            # Truncate the sentence to the maximum length of 512 tokens
            if len(sentence) > 512:
                sentence = sentence[:512]
            annotations = nlp(sentence)
            dict = {'pmcid' : pmcid, 'sentence_index' : index, 'n_male' : None, 'n_fem' : None, 'perc_male' : None, 'perc_fem' : None, 'sample' : None}
            for annotation in annotations:
                dict[annotation["entity"]] = json.dumps(annotation["word"])
            # If there were results from the model, add to results array
            if not ((dict['n_male'] is None) and (dict['n_male'] is None) and (dict['n_fem'] is None) and (dict['perc_male'] is None) and (dict['perc_fem'] is None) and (dict['sample'] is None)):
                values = list(dict.values())
                results.append(tuple(values))
            index += 1
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
    DB_FILE = config_all["api_europepmc_params"]["db_info_articles"]
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    # Create results table (if not already created)
    create_results_table(conn, cur)
    print("Created results table")

    # Get entries from db that need to be run through the model
    entries = get_entries(conn, cur)
    print("Got entries to be processed: ", len(entries))

    # Run entries through the model (sentence by sentence? check this)
    #nlp = pipeline("ner", model=args.model, device=0)
    nlp = pipeline("ner", model=args.model) # if you are working locally, remove device=0

    executor = concurrent.futures.ThreadPoolExecutor()
    futures = [
        executor.submit(run_model_on_entry, row, nlp) for row in entries
    ]

    pbar = tqdm(total=len(entries))
    for future in concurrent.futures.as_completed(futures)[0:20]:
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