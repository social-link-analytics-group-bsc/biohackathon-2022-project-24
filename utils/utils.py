import yaml
import logging

logger = logging.getLogger(
    __name__
)  ## Supposed to be a global logger to work in concurrent.futures
logging.basicConfig(level=logging.INFO)

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
