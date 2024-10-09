import sys
import os
import argparse
import yaml
import duckdb
import logging
import json
from tqdm import tqdm
from datasets import DatasetDict
from utils.model_loader import ModelLoader
from utils.post_process_answer import format_answer
from utils.preprocess_dataset import process_dataset, print_simple_info
from utils.utils import (
    dynamic_import,
    load_config,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def setup_logger() -> logging.Logger:
    """Setup the logger configuration for consistency."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


logger = setup_logger()


def recording_json_to_db(conn, json_response):
    # Record the json into a new table and add in the status the state of recording
    pass

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the script."""
    parser = argparse.ArgumentParser(description="Model Evaluation Script")

    parser.add_argument(
        "--model", required=True, type=str, help="Path or name of the model to load."
    )
    parser.add_argument(
        "--quantization", type=str, help="Quantization level for the model."
    )
    parser.add_argument(
        "--instruct",
        action="store_true",
        help="Flag indicating if the model is an instruct model.",
    )
    parser.add_argument(
        "--adapter",
        action="store_true",
        help="Flag to indicate if an adapter should be loaded.",
    )
    parser.add_argument(
        "--adapter_quantization",
        default=False,
        type=str,
        help="Specify if the adapter is quantized.",
    )
    parser.add_argument(
        "--prompt", required=True, type=str, help="Path to the prompt file."
    )
    return parser.parse_args()


def get_count_text(conn, table_sections):
    """
    Get the counts from the query
    """
    cursor = conn.cursor()
    query = f"SELECT COUNTS(*) FROM {table_sections} WHERE METHODS IS NOT NULL OR SUBJECTS IS NOT NULL"
    cursor.execute(query)
    count = cursor.fetchone()[0]
    cursor.close()
    return count

def get_text_from_db(conn, table_sections):
    """
    Get the already recorded pmcids from db
    """
    # Execute a SELECT query to retrieve the pmcid values
    # Fetch rows with abstract subjects and methods
    # Only return the document if methods or subjects are present.
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT ABSTRACT, SUBJECTS, METHODS FROM {table_sections} WHERE METHODS IS NOT NULL OR SUBJECTS IS NOT NULL"
    )
    row = cursor.fetchone()
    # Yield data if either subjects or methods are present
    # FIXME: Not working still return if None in the methods
    while row:
        if row[0]:
          yield row[0]  # Yield subjects
        elif row[1]:
            yield row[1]  # Yield methods
        else:
            yield None
        row = cursor.fetchone()
    # Close the cursor and connection
    cursor.close()
    conn.close()


def get_text_from_dataset(file_path):
    pass


def main():
    args = parse_arguments()
    logger.info(f"Arguments: {args}")

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "../config", "config.yaml")
    config_all = load_config(config_path)
    logger.info(f"Using model: {args.model}")
 
    # DB connection
    # # Name of the database
    DB_FILE = config_all["api_europepmc_params"]["db_info_articles"]
    table_status = config_all["db_params"]["table_status"]
    table_sections = config_all["db_params"]["table_sections"]
    table_metadata = config_all["db_params"]["table_metadata"]
    table_inference = config_all["db_params"]["table_inference"]
    # # Using duckdb to access the sqlite file for compatibility on marenostrum
    conn = duckdb.connect(DB_FILE)

    # Initialize ModelLoader
    model_loader = ModelLoader(
        model_path=args.model,
        quantization=args.quantization,
        instruct=args.instruct,
        adapter=args.adapter,
        adapter_quantization=args.adapter_quantization,
        generation_params=config_all["llm_params"]["generation_params"],
        bits_and_bytes_config=config_all["llm_params"]["bits_and_bytes"],
    )
    # Load the model
    llm_model = model_loader.get_model()

    # Load prompt instruction
    prompt_instruction = dynamic_import(f"utils.{args.prompt}", "prompt_instruction")

    # Parsing the articles
    total_articles = get_count_text(conn, table_sections ):
    n = 0
    for article_text in tqdm(get_text_from_db(conn, table_sections), total=total_articles):
        answer = llm_model.passing_article_to_llm(prompt_instruction=prompt_instruction, text=article_text,)
        
        
        print("ANSWER:\n")
        print(answer)
        n += 1
        if n == 10:
            raise


if __name__ == "__main__":
    main()
