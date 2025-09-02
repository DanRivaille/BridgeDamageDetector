#!/bin/bash
#SBATCH --job-name=ComputingMetrics
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

mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/ga1week_10-proportion/base_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/ga1week_10-proportion/fine-tuned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/ga1week_10-proportion/pruned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained

MODEL_ID=AE512_256x128
CONFIG_TYPE=month
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_256x128
CONFIG_TYPE=weekly
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_128x64
CONFIG_TYPE=month
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_128x64
CONFIG_TYPE=weekly
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_64x32
CONFIG_TYPE=month
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_64x32
CONFIG_TYPE=weekly
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/base_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/ga1week_10-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/fine-tuned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/ga1week_10-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/ga1week_10-proportion

mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/ga1week_20-proportion/base_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/ga1week_20-proportion/fine-tuned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/ga1week_20-proportion/pruned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained

MODEL_ID=AE512_256x128
CONFIG_TYPE=month
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_256x128
CONFIG_TYPE=weekly
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_128x64
CONFIG_TYPE=month
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_128x64
CONFIG_TYPE=weekly
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_64x32
CONFIG_TYPE=month
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_64x32
CONFIG_TYPE=weekly
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/base_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/ga1week_20-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/fine-tuned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/ga1week_20-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/ga1week_20-proportion

mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/ga1week_70-proportion/base_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/ga1week_70-proportion/fine-tuned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/ga1week_70-proportion/pruned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained

MODEL_ID=AE512_256x128
CONFIG_TYPE=month
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_256x128
CONFIG_TYPE=weekly
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_128x64
CONFIG_TYPE=month
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_128x64
CONFIG_TYPE=weekly
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_64x32
CONFIG_TYPE=month
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_64x32
CONFIG_TYPE=weekly
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/base_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/ga1week_70-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/fine-tuned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/ga1week_70-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/ga1week_70-proportion

mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/ga1week_80-proportion/base_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/ga1week_80-proportion/fine-tuned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/ga1week_80-proportion/pruned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained

MODEL_ID=AE512_256x128
CONFIG_TYPE=month
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_256x128
CONFIG_TYPE=weekly
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_128x64
CONFIG_TYPE=month
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_128x64
CONFIG_TYPE=weekly
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_64x32
CONFIG_TYPE=month
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_64x32
CONFIG_TYPE=weekly
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/base_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/ga1week_80-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/fine-tuned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/ga1week_80-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/ga1week_80-proportion

mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/ga1week_90-proportion/base_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/ga1week_90-proportion/fine-tuned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/ga1week_90-proportion/pruned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained

MODEL_ID=AE512_256x128
CONFIG_TYPE=month
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_256x128
CONFIG_TYPE=weekly
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_128x64
CONFIG_TYPE=month
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_128x64
CONFIG_TYPE=weekly
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_64x32
CONFIG_TYPE=month
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_64x32
CONFIG_TYPE=weekly
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/base_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/ga1week_90-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/fine-tuned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/ga1week_90-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/ga1week_90-proportion

mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_10-proportion/base_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_10-proportion/fine-tuned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_10-proportion/pruned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained

MODEL_ID=AE512_256x128
CONFIG_TYPE=month
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_256x128
CONFIG_TYPE=weekly
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_128x64
CONFIG_TYPE=month
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_128x64
CONFIG_TYPE=weekly
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_64x32
CONFIG_TYPE=month
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_64x32
CONFIG_TYPE=weekly
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/base_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_10-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/fine-tuned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_10-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_10-proportion

mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_20-proportion/base_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_20-proportion/fine-tuned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_20-proportion/pruned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained

MODEL_ID=AE512_256x128
CONFIG_TYPE=month
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_256x128
CONFIG_TYPE=weekly
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_128x64
CONFIG_TYPE=month
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_128x64
CONFIG_TYPE=weekly
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_64x32
CONFIG_TYPE=month
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_64x32
CONFIG_TYPE=weekly
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/base_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_20-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/fine-tuned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_20-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_20-proportion

mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_40-proportion/base_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_40-proportion/fine-tuned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_40-proportion/pruned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained

MODEL_ID=AE512_256x128
CONFIG_TYPE=month
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_256x128
CONFIG_TYPE=weekly
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_128x64
CONFIG_TYPE=month
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_128x64
CONFIG_TYPE=weekly
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_64x32
CONFIG_TYPE=month
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

MODEL_ID=AE512_64x32
CONFIG_TYPE=weekly
python3 $ROOT_PROJECT_FOLDER/Finetuning.py --id $MODEL_ID --config config_fune-tuning_model-$CONFIG_TYPE.json --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder base_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder pruned_models --save
python3 $ROOT_PROJECT_FOLDER/test.py --id $MODEL_ID --config config_compute_metrics-$CONFIG_TYPE.json --output-folder fine-tuned_models --save

mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/base_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_40-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/fine-tuned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_40-proportion
mv /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruned_models /home/ivan.santos/repositories/BridgeDamageDetector/models_trained/pruning_40-proportion

conda deactivate
