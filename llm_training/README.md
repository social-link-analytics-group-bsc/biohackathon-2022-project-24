# Running llm inference on Marenostrum

## Preparing on the Marenostrum

### Installing environment
#### Preparation steps

Before running the model it is needed to install the dependencies and downloading the models before connecting to an ACC node.

1. Connect to transfer3.bsc.es

```bash
ssh transfer3.bsc.es
```
2. copy your local git folder to your userfolder on Marenostrum

```bash
scp git ./$FOLDER
```
Once theses steps are done, you can log out from the `transfer3.bsc.es` node and connect to the ACC node `alogin4.bsc.es`

3. Install libraries
To install the library, the different modules are needed

```bash
cd ./$FOLDER
module purge
module load mkl intel impi hdf5 python cuda
# Equal to module load mkl/2024.0 intel/2024.0.1-sycl_cuda impi/2021.11 hdf5/1.14.1-2 python/3.12.1 cuda/12.2` at time of writing 2024-10-18
python -m venv venv
source venv/bin/activate
pip install -U langchain duckdb transformers accelerate peft datasets bitsandbytes rapidfuzz
```

Alternatively to installing packages manually

```
pip install -U -r requirements.txt
```

4. Downloading the model
Before using huggingface-cli to download the model it is needed to get an API Token as it will be asked for authentication in case the access to the model is restricted. 
To generate a token follow the guide here [https://huggingface.co/docs/hub/en/security-tokens](https://huggingface.co/docs/hub/en/security-tokens) and remember you can only see it once. 
Once it is done you can go to the models's folder and download the model you want. 
You need to use the `--cache-dir` option otherwise you will run out of space for caching the model during the download

```bash
cd llm_models/
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 --local-dir ./mixtral_7b_instruct_v03 --cache-dir ./cache 
```

5. Dataset preparation

The data are not in the github repository. You need to copy your local data in the data folder.
There are 2 different datasets you need depending if you want to evaluate the model and the prompt or running on the full dataset. 
To run the evaluation, the dataset is in jsonl format. The entire dataset shoudl be a sqlite3 file containing the entire downloaded articles from pubmed

6. Installing sql plugin for duckdb
In order for duckdb to work with sqlite3 the plugin sql needs to be installed.
Simply running the scrip `../duckdb_sqlite_installation.py` suffices.


### Activate environment

```bash
module purge &&
module load  mkl/2024.0 nvidia-hpc-sdk/23.11-cuda11.8 openblas/0.3.27-gcc cudnn/9.0.0-cuda11 tensorrt/10.0.0-cuda11 impi/2021.11 hdf5/1.14.1-2-gcc gcc/11.4.0 python/3.11.5-gcc nccl/2.19.4 pytorch
source venv/bin/activate
```

## Running the script

### Evaluation
The script `evaluator.py`, is designed to evaluate a language model parameters such as quantization, instruct modes, and adapter support. 
The script allows for flexible evaluation by accepting a range of command-line arguments.

#### Usage
To run the evaluation script, use the following basic command:

```bash
    python evaluator.py --model <model_name_or_path> --prompt <prompt_file_path> --training_set <training_set_path> [options]
```
##### Command-Line Arguments

Here are the available command-line arguments for running the script:

--model (required): The path to the model.

--prompt (required): The name of the prompt file that contains the input prompts for the model. The prompt are stored in python file in `./utils`. The name should omit the `.py` extension (*i.e.: prompt2*).

--training_set (required): The path to the training dataset for evaluation. It is on `jsonl`format and directly from prodigy. The process to convert it to meaningful text is taken care by `./utils/preprocess_dataset.py`.

--quantization (optional): Specifies the quantization level for the model, if applicable. Options are `4bits` and `8bits`
> [!IMPORTANT]
> The 8bits is not working right now due to a bug in the combination of pytorch and cuda version

--instruct (optional): A flag indicating whether the model is an instruct model. Use this flag if the model supports instruction-based prompting.

--adapter (optional): A flag to indicate whether an adapter should be loaded for the model.

--adapter_quantization (optional): Specifies if the adapter is quantized. Defaults to False.

--full_eval (optional): A flag to indicate whether to run the evaluation on the entire dataset, without splitting to 80/20.


#### Example Commands

1. Basic Usage:

```bash
python evaluator.py --model my_model --prompt prompt1 --training_set data/training_data/final_annotations_XXX.jsonl --instruct --adapter --quantization 4bits 
```

1. Using Instruct Mode:

```bash

python evaluator.py --model my_model --prompt prompt1 --training_set data/training_data/final_annotations_XXX.jsonl --instruct
```

1. With Quantization and Adapter:

```bash
python evaluator.py --model my_model --prompt prompt1 --training_set data/training_data/final_annotations_XXX.jsonl --instruct --adapter --quantization 4bits 
```

1. Full Dataset Evaluation:

```bash
python evaluator.py --model my_model --prompt prompt1 --training_set data/training_data/final_annotations_XXX.jsonl --full_eval
```


### Inference
