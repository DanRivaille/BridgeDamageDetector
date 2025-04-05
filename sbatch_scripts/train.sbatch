#!/bin/bash
#SBATCH --job-name=RunFramework
#SBATCH --output=/scratch/ivan.santos/test_models/output/train_and_test.out
#SBATCH --error=/scratch/ivan.santos/test_models/error/train_and_test.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=10:00:00
#SBATCH --partition GPU

ROOT_PROJECT_FOLDER=/home/ivan.santos/repositories/BridgeDamageDetector
SOURCE_FOLDER=$ROOT_PROJECT_FOLDER/src

MODEL_ID=AE512_64x32
CONFIG_TYPE=weekly

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate AE-GA

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-$CONFIG_TYPE.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $SOURCE_FOLDER/reduce_model/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $SOURCE_FOLDER/reduce_model/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

conda deactivate
