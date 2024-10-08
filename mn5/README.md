# Installing and configuring environment
## Loading modules
```bash
module purge && 
module load  mkl/2024.0 nvidia-hpc-sdk/23.11-cuda11.8 openblas/0.3.27-gcc cudnn/9.0.0-cuda11 tensorrt/10.0.0-cuda11 impi/2021.11 hdf5/1.14.1-2-gcc gcc/11.4.0 python/3.11.5-gcc nccl/2.19.4 pytorch ncurses tmux &&
```
## VirtualEnv
```
cd /gpfs/projects/bsc02/sla_projects/biohack-2022/
```
```
source ./venv_311/bin/activate
```
```
pip install -r requirements_311.txt
```
## Downloading the model
```
cd llm_models/
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 --local-dir ./mixtral_8x7b_instruct_v03 --cache-dir ./cache 
```
# Running the script

## Training
```
accelerate launch --config_file ./config/fsdp_config_qlora.yaml llm_inference/run_fsdp_qlora.py --config config/llama_3_70b_fsdp_qlora.yaml
```
## Evaluation
```
python llm_inference/llm_evaluate_models.py --model config/$config_model  --quantization [4bits,8bits,None] --adapter $BOOL --adapter_quantization [4bits,8bits,None]
```
## Inference

