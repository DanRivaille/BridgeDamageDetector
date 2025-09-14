#!/bin/bash
#SBATCH --job-name=RunFramework
#SBATCH --output=/scratch/ivan.santos/test_models/output/train_and_test.out
#SBATCH --error=/scratch/ivan.santos/test_models/error/train_and_test.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4096
#SBATCH --partition GPU
#SBATCH --qos=normal-gpu

ROOT_PROJECT_FOLDER=/home/ivan.santos/repositories/BridgeDamageDetector

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate AE-GA

MODEL_ID=AE512_256x128
CONFIG_TYPE=month

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_90.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

CONFIG_TYPE=weekly

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_90.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_128x64
CONFIG_TYPE=month

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_90.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

CONFIG_TYPE=weekly

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_90.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_64x32
CONFIG_TYPE=month

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_90.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

CONFIG_TYPE=weekly

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_90.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

mkdir /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_90-proportion

mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/base_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_90-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/fine-tuned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_90-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_90-proportion


MODEL_ID=AE512_256x128
CONFIG_TYPE=month

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_80.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

CONFIG_TYPE=weekly

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_80.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_128x64
CONFIG_TYPE=month

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_80.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

CONFIG_TYPE=weekly

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_80.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_64x32
CONFIG_TYPE=month

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_80.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

CONFIG_TYPE=weekly

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_80.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

mkdir /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_80-proportion

mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/base_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_80-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/fine-tuned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_80-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_80-proportion

MODEL_ID=AE512_256x128
CONFIG_TYPE=month

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_70.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

CONFIG_TYPE=weekly

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_70.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_128x64
CONFIG_TYPE=month

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_70.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

CONFIG_TYPE=weekly

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_70.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_64x32
CONFIG_TYPE=month

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_70.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

CONFIG_TYPE=weekly

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_70.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

mkdir /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_70-proportion

mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/base_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_70-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/fine-tuned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_70-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_70-proportion

MODEL_ID=AE512_256x128
CONFIG_TYPE=month

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_60.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

CONFIG_TYPE=weekly

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_60.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_128x64
CONFIG_TYPE=month

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_60.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

CONFIG_TYPE=weekly

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_60.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_64x32
CONFIG_TYPE=month

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_60.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

CONFIG_TYPE=weekly

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_60.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

mkdir /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_60-proportion

mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/base_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_60-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/fine-tuned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_60-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_60-proportion

MODEL_ID=AE512_256x128
CONFIG_TYPE=month

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_50.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

CONFIG_TYPE=weekly

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_50.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_128x64
CONFIG_TYPE=month

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_50.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

CONFIG_TYPE=weekly

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_50.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_64x32
CONFIG_TYPE=month

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_50.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

CONFIG_TYPE=weekly

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_50.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

mkdir /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_50-proportion

mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/base_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_50-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/fine-tuned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_50-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_50-proportion

MODEL_ID=AE512_256x128
CONFIG_TYPE=month

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_40.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

CONFIG_TYPE=weekly

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_40.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_128x64
CONFIG_TYPE=month

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_40.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

CONFIG_TYPE=weekly

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_40.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_64x32
CONFIG_TYPE=month

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_40.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

CONFIG_TYPE=weekly

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_40.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

mkdir /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_40-proportion

mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/base_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_40-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/fine-tuned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_40-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_40-proportion

MODEL_ID=AE512_256x128
CONFIG_TYPE=month

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_30.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

CONFIG_TYPE=weekly

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_30.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_128x64
CONFIG_TYPE=month

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_30.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

CONFIG_TYPE=weekly

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_30.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_64x32
CONFIG_TYPE=month

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_30.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

CONFIG_TYPE=weekly

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_30.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

mkdir /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_30-proportion

mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/base_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_30-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/fine-tuned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_30-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_30-proportion

MODEL_ID=AE512_256x128
CONFIG_TYPE=month

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_20.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

CONFIG_TYPE=weekly

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_20.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_128x64
CONFIG_TYPE=month

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_20.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

CONFIG_TYPE=weekly

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_20.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_64x32
CONFIG_TYPE=month

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_20.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

CONFIG_TYPE=weekly

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_20.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

mkdir /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_20-proportion

mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/base_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_20-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/fine-tuned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_20-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_20-proportion

MODEL_ID=AE512_256x128
CONFIG_TYPE=month

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_10.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

CONFIG_TYPE=weekly

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_10.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_128x64
CONFIG_TYPE=month

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_10.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

CONFIG_TYPE=weekly

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_10.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_64x32
CONFIG_TYPE=month

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_10.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

CONFIG_TYPE=weekly

# Train the base model
python3 $ROOT_PROJECT_FOLDER/train.py --id $MODEL_ID --config config_train_base_model-$CONFIG_TYPE.json --output-folder base_models --save

# Run the optimization technique to find the best mask
python3 $ROOT_PROJECT_FOLDER/ga_bridge.py --id $MODEL_ID --config config_ga_optimization-${CONFIG_TYPE}_10.json --output-folder base_models --save

# Reimplement the model using the mask
python3 $ROOT_PROJECT_FOLDER/Reduce.py --id $MODEL_ID --config config_reduce_model-$CONFIG_TYPE.json --output-folder base_models --save

# Fine-tuning the reimplemented model
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save

# Get the metrics of three models (base, reimplemented and fine-tuned)
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

mkdir /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_10-proportion

mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/base_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_10-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/fine-tuned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_10-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_10-proportion

conda deactivate
