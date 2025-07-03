import os
import ast
import sys
import yaml
import json
import logging
import logging
import torch
import datasets
from peft import PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from utils.prompt_instructions import prompt_instruction_3, json_response_format


logger = logging.getLogger(
    __name__
)  ## Supposed to be a global logger to work in concurrent.futures
logging.basicConfig(level=logging.INFO)


class LLMHandler:
    def __init__(
        self,
        model_path,
        generation_params,
        prompt_instruction=None,
        bits_and_bytes_config=None,
        adapter_path=None,
        # torchtype=torch.bfloat16,
        torchtype=torch.float,
    ):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "auto"
        self.torchtype = torchtype
        self._print_state()
        self.prompt_instruction = prompt_instruction
        self.generation_params = generation_params
        self._generate_bits_and_bytes(bits_and_bytes_config)
        self._load_model(model_path, adapter_path)
        self._print_state()

    def _generate_bits_and_bytes(self, quantization_config):

        if quantization_config:

            # raise Exception("Need to fix that to include the version of 8bits")
            quantization_config["bnb_4bit_compute_dtype"] = torch.bfloat16
            self.quantization = BitsAndBytesConfig(**quantization_config)
        else:
            self.quantization = None

    def _load_model(self, model_path, adapter_path):
        logger.info(
            f"Load: {model_path} - adapter_path: {adapter_path} - quantisation: {self.quantization}"
        )
        # Load the Lora model

        if adapter_path:
            config = PeftConfig.from_pretrained(adapter_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                torch_dtype=self.torchtype,
                attn_implementation="sdpa",  # use sdpa, alternatively use "flash_attention_2"
                quantization_config=self.quantization,
                device_map=self.device,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.base_model_name_or_path
            )

            # self.tokenizer.pad_token = self.tokenizer.eos_token  # Most LLMs don't have a pad token by default
            # self.model = PeftModel.from_pretrained(self.model, adapter_path)
        # If no adapter_path load base model
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                attn_implementation="sdpa",  # use sdpa, alternatively use "flash_attention_2"
                torch_dtype=self.torchtype,
                # low_cpu_mem_usage=True,
                device_map=self.device,
                quantization_config=self.quantization,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            # self.tokenizer.pad_token = self.tokenizer.eos_token  # Most LLMs don't have a pad token by default

    def _print_state(self):
        """
        Print the device information
        """
        # Check cuda
        print("Using self.device:", self.device)
        print()

        # Additional Info when using cuda
        try:
            if self.device.type == "cuda":
                print(torch.cuda.get_device_name(0))
                print("Memory Usage:")
                print(
                    "Allocated:",
                    round(torch.cuda.memory_allocated(0) / 1024**3, 1),
                    "GB",
                )
                print(
                    "Cached:   ",
                    round(torch.cuda.memory_reserved(0) / 1024**3, 1),
                    "GB",
                )
        except AttributeError:  # In case of auton
            pass

    def _check_prompt_instruction(self, prompt_instruction):
        if self.prompt_instruction is None:
            if prompt_instruction:
                return prompt_instruction
            else:
                raise Exception("No prompt instruction message given")
        if self.prompt_instruction:
            if prompt_instruction is None:
                return self.prompt_instruction
            else:
                raise Exception(
                    "Passed instruction in the init and in the method, need to remove one"
                )

    def construct_prompt(self, prompt_instruction, text):
        return f"{prompt_instruction}\n{text}"

    def encode_input(self, message):
        inputs = self.tokenizer(
            message, return_tensors="pt", add_generation_prompt=False
        ).to(self.device)
        return inputs

    def generate_output(self, encoded):
        generation_output = self.model.generate(**encoded, **self.generation_params)
        return generation_output

    # def decode_output(self, inputs, output):
    #     return self.tokenizer.decode(output[0, len(inputs):])
    #     # text_output = self.tokenizer.decode(token_output, skip_special_tokens=True)
    #     # return text_output

    def retrieve_answer(self, inputs, outputs):

        prompt_length = inputs.shape[1]

        return self.tokenizer.decode(
            outputs[0][prompt_length:], skip_special_tokens=True
        )

    def passing_article_to_llm(self, text, prompt_instruction=None):
        prompt_instruction = self._check_prompt_instruction(prompt_instruction)
        prompt_message = self.construct_prompt(prompt_instruction, text)

        inputs = self.encode_input(prompt_message)
        output = self.generate_output(inputs)
        return self.retrieve_answer(inputs, output)


class LLMHandlerInstruct(LLMHandler):

    def generate_output(self, encoded):
        generation_output = self.model.generate(encoded, **self.generation_params)
        return generation_output

    def encode_input(self, message):
        messages = [{"role": "user", "content": message}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            add_special_tokens=True,
        )
        # .to(self.device)
        return inputs


def main():

    # Load the config path
    config_path = os.path.join(os.path.dirname(__file__), "../config", "config.yaml")
    config_all = yaml.safe_load(open(config_path))

    logger.info("Load the data for evaluation")
    # Load the data from the model prediction
    training_set_path = config_all["llm_params"]["training_set_path"]
    data_loaded = datasets.load_from_disk(training_set_path)
    logger.info("Data loadaed")

    # LLM setting
    model_outdir = config_all["llm_params"]["model_outdir"]
    model_name = config_all["llm_params"]["model_name"]
    model_path = f"{model_outdir}/{model_name}"
    instruct_model = config_all["llm_params"]["instruct_model"]
    # Get the adapter
    try:
        adapter_name = config_all["llm_params"]["adapter"]
        adapter_path = f"{model_outdir}/{adapter_name}"
    except KeyError:
        adapter_path = None
    generation_params = config_all["llm_params"]["generation_params"]
    # Load bitsandBytes config
    try:
        bits_and_bytes_config = config_all["llm_params"]["bits_and_bytes_config"]
    except KeyError:
        bits_and_bytes_config = None

    # Instantiate the model
    logger.info("Load the model in the GPU(s)")
    if instruct_model:
        llm_model = LLMHandlerInstruct(
            model_path,
            generation_params,
            bits_and_bytes_config=bits_and_bytes_config,
            adapter_path=adapter_path,
        )
    else:
        llm_model = LLMHandler(
            model_path,
            generation_params,
            bits_and_bytes_config=bits_and_bytes_config,
            adapter_path=adapter_path,
        )
    logger.info("Model loaded")
    methods = [
        "There is 4 women and 3 men for a total of 7 subjects",
        "Only 1 woman agreed",
        "Among the 12 mice, only 4 were alive at the end of the experiment",
    ]  # Define your method section text
    # text_output, prompt_text, answer = llm_handler.passing_article_to_llm(methods)
    for example in methods:
        text_output = llm_model.passing_article_to_llm(
            example, prompt_instruction=prompt_instruction_3
        )
        # print("PROMPT CONTEXT:\n")
        # print(prompt_text)
        # print("\n")
        # print("GENERATED ANSWER:\n")
        # print(answer)
        # print("\n")
        print("TEXT OUTPUT:\n")
        print(text_output)


if __name__ == "__main__":
    main()
