# model_loader.py

import logging
from typing import Optional, Tuple

# from utils.handlers import LLMHandler, LLMHandlerInstruct
from utils.inference_class import LLMHandler, LLMHandlerInstruct

from utils.utils import setup_bits_and_bytes_config, setup_adapter_path


class ModelLoader:
    """
    A class to handle loading of language models with optional quantization and adapter settings.
    """

    def __init__(
        self,
        model_path: str,
        quantization: Optional[str],
        instruct: bool,
        adapter: bool,
        adapter_quantization: Optional[str],
        generation_params: dict,
        bits_and_bytes_config: dict,
    ):
        """
        Initializes the ModelLoader with necessary configurations.

        Args:
            model_path (str): Path or name of the model to load.
            quantization (Optional[str]): Quantization level for the model.
            instruct (bool): Flag indicating if the model is an instruct model.
            adapter (bool): Flag to indicate if an adapter should be loaded.
            adapter_quantization (Optional[str]): Specify if the adapter is quantized.
            generation_params (dict): Parameters for text generation.
            bits_and_bytes_config (dict): Configuration for bits and bytes.
        """
        self.model_path = model_path
        self.quantization = quantization
        self.instruct = instruct
        self.adapter = adapter
        self.adapter_quantization = adapter_quantization
        self.generation_params = generation_params
        self.bits_and_bytes_config = bits_and_bytes_config
        self.adapter_path: Optional[str] = None
        self.model: Optional[object] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    def configure_adapter(self) -> None:
        """Sets up the adapter path based on the provided configurations."""
        self.logger.info("Configuring adapter path...")
        self.adapter_path = setup_adapter_path(
            self.model_path, self.adapter, self.adapter_quantization
        )
        self.logger.info(f"Adapter path set to: {self.adapter_path}")

    def configure_bits_and_bytes(self) -> Tuple[Optional[str], dict]:
        """
        Sets up bits and bytes configuration based on quantization settings.

        Returns:
            Tuple containing the model quantization level and bits-and-bytes configuration.
        """
        self.logger.info("Configuring BitsAndBytes...")
        self.model_quantization, self.bits_and_bytes = setup_bits_and_bytes_config(
            self.quantization, self.bits_and_bytes_config
        )
        self.logger.info(f"BitsAndBytes Config: {self.bits_and_bytes}")
        return self.model_quantization, self.bits_and_bytes

    def load_model(self) -> object:
        """
        Loads the model using the appropriate handler.

        Returns:
            The loaded model object.
        """
        self.logger.info("Loading model...")
        self.configure_bits_and_bytes()
        self.configure_adapter()

        model_class = LLMHandlerInstruct if self.instruct else LLMHandler
        self.model = model_class(
            self.model_path,
            self.generation_params,
            bits_and_bytes_config=self.bits_and_bytes,
            adapter_path=self.adapter_path,
        )
        self.logger.info("Model loaded successfully.")
        return self.model

    def get_model(self) -> object:
        """
        Returns the loaded model, loading it if not already loaded.

        Returns:
            The loaded model object.
        """
        if not self.model:
            self.load_model()
        return self.model
