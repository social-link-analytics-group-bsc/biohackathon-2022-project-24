# Standard library imports
import argparse
import ast
import datetime
import io
import json
import logging
import os
import random
import re
import sys
import functools 
from operator import add
from pprint import pprint

# Third-party library imports
import duckdb
import numpy as np
import tqdm
from IPython.display import Image, display 
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from typing import Annotated, Any, Callable, Dict, List, Literal, Optional, Union
from typing_extensions import Literal as TypingExtensionsLiteral, TypedDict # Use alias to avoid conflict with typing.Literal

# Local application/library specific imports
from utils.utils import dynamic_import, load_config

def setup_logger() -> logging.Logger:
    """Setup the logger configuration for consistency."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)





def main():
    logger = setup_logger()
    # Set to DEBUG for more detailed output:
    logger.setLevel(logging.DEBUG)  # Set the level for your logger instance
    # Load config
    config_path = os.path.join("../config", "config.yaml")
    config_all = load_config(config_path)
    # DB connection
    # # Name of the database
    DB_FILE = config_all["api_europepmc_params"]["db_info_articles"]
    table_status = config_all["db_params"]["table_status"]
    table_sections = config_all["db_params"]["table_sections"]
    table_metadata = config_all["db_params"]["table_metadata"]
    table_inference = config_all["db_params"]["table_inference"]
    BSC = config_all["agent_params"]["use_bsc"]

    if BSC:
        running_location = 'bsc'
    else:
        running_location = 'local'

    llm_model_name = config_all["agent_params"]['llm_model'][running_location]["name"]
    llm_connection_port = config_all['agent_params']['llm_model'][running_location]['port']

    llm_connection_host = config_all['agent_params']['llm_connection']["host"]
    llm_connection_api_end = config_all['agent_params']['llm_connection']['api_end']
    llm_connection_api_key = config_all['agent_params']['llm_connection']['api_key']

    open_ai_url = f"{llm_connection_host}:{llm_connection_port}/{llm_connection_api_end}"

    test_decision_json = config_all['agent_params']["test_decision_json"]





