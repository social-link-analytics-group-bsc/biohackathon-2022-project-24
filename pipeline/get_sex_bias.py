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
nltk.download('punkt')  # Download the sentence tokenizer model (if not already downloaded)
from nltk.tokenize import sent_tokenize


def parsing_arguments(parser):
    parser.add_argument("--model", type=str, default='output/bert-base-uncased-en/sbe.py_8_0.00005_date_22-11-10_time_14-55-26',
                        help='Pretrained model to find the numbers')
    return parser

# Method for getting database entries that still need to be run through the model
def get_entries(conn):
    SQL_QUERY = 'SELECT Main.pmcid, Main.methods FROM Main LEFT JOIN Checks ON Main.pmcid = Checks.pmcid WHERE Checks.results IS NULL;'
    cur = conn.cursor()
    cur.execute(SQL_QUERY)
    return cur.fetchall()

# Create table to record model results in (pmcid, n_fem, n_male, per_fem, perc_male, sample)
def create_results_table(conn):
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS "Results" (
                "pmcid"	TEXT NOT NULL,
                "sentence_index"	INTEGER,
                "n_fem"	TEXT,
                "n_male"	TEXT,
                "perc_fem"	TEXT,
                "perc_male"	TEXT,
                "sample"	TEXT,
                FOREIGN KEY ("pmcid") REFERENCES Main("pmcid")
            )"""
        )
    conn.commit()

# Add new row to results table
def add_result(conn, pmcid, results):
    cur = conn.cursor()
    SQL_QUERY = "INSERT INTO Results (pmcid, sentence_index, n_male, n_fem, perc_male, perc_fem, sample) VALUES  (?, ?, ?, ?, ?, ?, ?)"
    for row in results:
        cur.execute(SQL_QUERY, row)

    SQL_QUERY = "INSERT OR REPLACE INTO Checks (pmcid, results) VALUES (?, 1);"
    cur.execute(SQL_QUERY, (pmcid,))
    conn.commit()

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

    # Create results table (if not already created)
    create_results_table(conn)
    print("Created results table")

    # Get entries from db that need to be run through the model
    entries = get_entries(conn)
    print("Got entries to be processed")

    # Run entries through the model (sentence by sentence? check this)
    #nlp = pipeline("ner", model=args.model, device=0)
    nlp = pipeline("ner", model=args.model) # if you are working locally, remove device=0
    for row in entries:

        pmcid, methods = row
        # Split methods section into sentences
        
        if methods is not None:
            sentences = sent_tokenize(methods)
            # Establish array of tuples of results per sentence
            results = []
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

            add_result(conn, pmcid, results)
        else:
            continue

    conn.close()


if __name__ == "__main__":
    main()