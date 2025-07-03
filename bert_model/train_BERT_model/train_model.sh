#!/bin/bash
#SBATCH --job-name="Train model"
#SBATCH --chdir=./
#SBATCH --output=./logs/sts_%j.out
#SBATCH --error=./logs/sts_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --qos=debug

$MODEL_NAME_OR_PATH=''
$OUTPUT_DIR=''
$NUM_TRAIN_EPOCHS=1

timestamp()
{
 date +"%Y-%m-%d %T"
}

PYTHON_VERSION=3.10

module purge && module load cmake gcc intel mkl python/$PYTHON_VERSION.2

echo "$(timestamp) Activate environment"
source ../venv/bin/activate

SCRIPT_DIR="./pipeline/train_model/"
cd $SCRIPT_DIR

echo "$(timestamp) Running the model job"
srun python$PYTHON_VERSION run_ner.py --model_name_or_path=$MODEL_NAME_OR_PATH --output_dir=$OUTPUT_DIR --do_train --do_eval --num_train_epochs=$NUM_TRAIN_EPOCHS

# TODO - Define more arguments: model, tokenizer (try Longformer) 
#                               dataset (sbe is already invoked somehow) 
# TODO - Make it run with 2 GPUs. Check whether it increases the efficiency and runs faster.


echo "$(timestamp) Job done!"
