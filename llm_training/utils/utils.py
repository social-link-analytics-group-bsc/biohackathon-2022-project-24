import os
import logging
import torch
import yaml
import importlib
from accelerate import PartialState

logger = logging.getLogger(
    __name__
)  ## Supposed to be a global logger to work in concurrent.futures
logging.basicConfig(level=logging.INFO)


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def dynamic_import(module_name: str, variable_name: str):
    """
    Dynamically imports a variable from a given module.

    Args:
        module_name (str): The full name of the module to import, e.g., 'utils.prompt1'.
        variable_name (str): The name of the variable to import from the module.

    Returns:
        Any: The imported variable if successful, or None if the import fails.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the variable does not exist in the module.
    """
    try:
        module = importlib.import_module(module_name)
        variable = getattr(module, variable_name)
        return variable
    except (ImportError, AttributeError) as e:
        raise Exception(f"Error: {e}")


def setup_adapter_path(model_path, adapter=False, quantization=False):
    logger.debug(
        f"Adapter: {adapter} - model path: {model_path} - quantization: {quantization}"
    )
    if adapter:
        if quantization:
            adapter_path = f"{model_path}_lora_{quantization}"
        else:
            adapter_path = f"{model_path}_lora"
    else:
        adapter_path = None
    return adapter_path


def setup_bits_and_bytes_config(quantization, config):
    bits_and_bytes_config = None
    if quantization:
        bits_and_bytes_config = config.get(quantization, None)
        if bits_and_bytes_config == None:
            raise Exception(
                f"Could not parse a bits_and_bytes_config from the file with the key: {quantization}.\nCurrent keys in the config: {config.keys()}"
            )
    logger.info(f"bitsandbytes config returned: {bits_and_bytes_config}")
    return quantization, bits_and_bytes_config


def print_cuda_state():
    """
    Print the device information
    """
    # Check cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Additional Info when using cuda
    if device.type == "cuda":
        for i in range(torch.cuda.device_count()):
            logging.info(torch.cuda.get_device_properties(i))
        logging.info(torch.cuda.get_device_name())
        logging.info("Memory Usage:")
        logging.info(
            f"Allocated: {round(torch.cuda.memory_allocated() / 1024**3, 1)} GB"
        )
        logging.info(f"Cached: {round(torch.cuda.memory_reserved() / 1024**3, 1)} GB")


if __name__ == "__main__":
    print_cuda_state()
    device_string = PartialState().process_index
    print(device_string)
