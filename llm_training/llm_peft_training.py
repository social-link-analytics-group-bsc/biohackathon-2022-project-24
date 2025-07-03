import argparse
import os
import sys
import logging
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import datasets
from datasets import DatasetDict
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
from peft import prepare_model_for_kbit_training

from utils.utils import (
    print_cuda_state,
    load_config,
    setup_model_path,
    setup_adapter_path,
)
import deepspeed
from trl import SFTTrainer

from transformers.deepspeed import HfDeepSpeedConfig

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(
    __name__
)  ## Supposed to be a global logger to work in concurrent.futures
logging.basicConfig(level=logging.DEBUG)


def return_tokenizer(model_path, training_set_path):

    # FIXME Need to move that into the prep dataset to ensure reproductiblity (but the seed should be enough)
    ds_training_set = datasets.load_from_disk(training_set_path)
    # Split a first time to gain the train set and the test set
    ds_training_set = ds_training_set.train_test_split(
        test_size=0.2, seed=42, stratify_by_column="answer"
    )
    # Split the test set a second time to get the validation set and the test set
    ds_devtest = ds_training_set["test"].train_test_split(
        test_size=0.5, seed=42, stratify_by_column="answer"
    )
    # Create a new DatasetDict to get all of them in one place
    training_set = DatasetDict(
        {
            "training": ds_training_set["train"],
            "validation": ds_devtest["train"],
            "test": ds_devtest["test"],
        }
    )
    # Tokenizing the training set
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",  # Add padding as use less memory for training
        add_eos_token=True,
        add_bos_token=True,
    )
    # To still pad but not training on th pad_token but still on the eos_token
    # tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token = tokenizer.eos_token
    # Need to be 'input_ids' to be passed to a SFTTrainer trainer class
    tokenized_train_ds = training_set.map(
        lambda x: {
            "input_ids": tokenizer.apply_chat_template(
                x["message"], tokenize=True, add_generation_prompt=False
            )
        }
    )
    # Getting the max token length from the training set and adjust that as padding length
    max_length = max([len(x) for x in tokenized_train_ds["training"]["input_ids"]])
    print(max_length)

    # Retokenized with the max token length
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",  # Add padding as use less memory for training
        add_eos_token=True,
        add_bos_token=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

    # tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_train_ds = training_set.map(
        lambda x: {
            "input_ids": tokenizer.apply_chat_template(
                x["message"], tokenize=True, add_generation_prompt=False
            )
        }
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    return (
        tokenized_train_ds,
        data_collator,
        tokenized_train_ds["training"],
        tokenized_train_ds["test"],
    )


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Args to pass accelerate config file, set up the trainer into fsdp mode",
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Args automatically passed by deepspeed launcher in case of multi gpu",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Config file path for the deepspeed config file",
    )
    parser.add_argument(
        "--model",
        default=None,
        type=str,
        required=True,
        help="Give the config filename to the model to be run. ",
    )
    parser.add_argument(
        "--quantization",
        default=None,
        type=str,
        required=False,
        help="Decide the level of quantization for the model",
    )
    parser.add_argument(
        "--adapter",
        default=False,
        type=bool,
        required=False,
        help="To know if load a adapter or not",
    )
    return parser


def bits_and_bytes_config():
    pass


def main():
    # Init the deepspeed distributed
    # Need to load the config
    # deepspeed.init_distributed()
    # print_cuda_state()
    args = parser().parse_args()
    logger.info(f"Using model: {args.model}")
    config_path = os.path.join(os.path.dirname(__file__), "../config", "config.yaml")
    config_all = load_config(config_path)
    config_model_path = os.path.join(
        os.path.dirname(__file__), "../config", f"{args.model}.yaml"
    )
    config_model = load_config(config_model_path)

    model_path = setup_model_path(config_all, config_model)
    training_set_path = config_all["llm_params"]["training_set_path"]
    training_args = config_all["llm_params"]["trainer_params"]

    logger.info("Loading options")

    ###### DATASET AND TOKENISER
    tokenizer, data_collator, train_dataset, eval_dataset = return_tokenizer(
        model_path, training_set_path
    )
    model_quantization = False
    if args.quantization:
        model_quantization = args.quantization
    else:
        try:
            model_quantization = config_model["peft"]["quantization"]
        except KeyError:
            pass
    bitsandbytes = None
    if (
        model_quantization is not False
    ):  # Can be 4bits or 8bits to load appropriate bits-and-bytes config
        bitsandbytes = config_all["llm_params"]["bits_and_bytes"][model_quantization]
        if model_quantization == "8bits":
            bitsandbytes["bnb_8bit_compute_dtype"] = torch.bfloat16
        elif model_quantization == "4bits":
            bitsandbytes["bnb_4bit_compute_dtype"] = torch.bfloat16
        else:
            pass

    # As it is to peft-training there will be adapter True
    adapter_path, adapter_id = setup_adapter_path(
        model_path, model_quantization, adapter=True
    )

    deepspeed_config = args.deepspeed

    # LLM setting
    lora_config = config_all["llm_params"]["lora_config"]
    logger.info(
        f"Model quant: {model_quantization} - bitsandbytes: {bitsandbytes}\n####"
    )
    logger.info(f"lora config: {lora_config}\n####")

    # Load LORA config
    peft_config = LoraConfig(**lora_config)

    ####### # Instantiate the trainer
    # Instantiate the trainer
    config_trainers_params = config_all["llm_params"]["trainer_params"]
    training_args = TrainingArguments(
        output_dir=adapter_path,
        deepspeed=deepspeed_config,
        **config_trainers_params,
    )
    ##### MODEL LOADING WITH SFTT
    # Load the model with bitsandbytes
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bitsandbytes,
        torch_dtype=torch.bfloat16,  # Not sure it is needed as it is in the bitsandbytes config
        attn_implementation="flash_attention_2",
        use_cache=False,  # set to False as we're going to use gradient checkpointing
        low_cpu_mem_usage=False,  # Seems to be incompatible with zero3 but turn on True automatically if model quantized and not set up
        # device_map={"": torch.cuda.current_device()},
    )
    model.gradient_checkpointing_enable()

    ###### MODEL LOADING with Trainer

    # # Print the mem used
    logger.info("After model instanciate")
    # print_cuda_state()
    # # Prepare the model for peft training with the quantization params
    if model_quantization is not False:
        model = prepare_model_for_kbit_training(model)

    # Load the model in peft mode
    model = get_peft_model(model, peft_config)

    logger.info("After peft_model Instanciate")
    print_cuda_state()

    # Loading and preparing the training set
    logger.info("Data loading and tokenizing")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Running the training")
    trainer.train()
    logger.info("Training over")

    # Loading the LORA model
    logger.info("Loading the LORA model")
    lora_model = get_peft_model(model, peft_config)
    lora_model.print_trainable_parameters()

    logger.info("Saving the model")
    trainer.model.save_pretrained(adapter_path)


if __name__ == "__main__":
    main()
