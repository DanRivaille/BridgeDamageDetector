#!/bin/bash
#SBATCH --job-name=TrnTstML
#SBATCH --output=/scratch/ivan.santos/test_models/output/train_and_test.out
#SBATCH --error=/scratch/ivan.santos/test_models/error/train_and_test.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=10:00:00
#SBATCH --partition GPU

ROOT_PROJECT_FOLDER=/home/ivan.santos/repositories/AE-Z24
SOURCE_FOLDER=$ROOT_PROJECT_FOLDER/src

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate AE-Z24

#python3 $ROOT_PROJECT_FOLDER/train.py --id example --config $ROOT_PROJECT_FOLDER/config_files/config_example.json --output-folder example-run --save
#python3 $ROOT_PROJECT_FOLDER/test.py --id example --config $ROOT_PROJECT_FOLDER/config_files/config_example.json --output-folder example-run --save

# PARTIAL LSTM-AE (SEQUENCES)
#python3 $ROOT_PROJECT_FOLDER/train.py --id sequence_test --config $ROOT_PROJECT_FOLDER/config_files/config_sequence_$SLURM_ARRAY_TASK_ID.json --output-folder sequences --save
#python3 $ROOT_PROJECT_FOLDER/test.py  --id sequence_test --config $ROOT_PROJECT_FOLDER/config_files/config_sequence_$SLURM_ARRAY_TASK_ID.json --output-folder sequences --save

# PARTIAL LSTM-AE (TOPOLOGIES)
#python3 $ROOT_PROJECT_FOLDER/train.py --id topology_test --config $ROOT_PROJECT_FOLDER/config_files/config_topology_$SLURM_ARRAY_TASK_ID.json --output-folder topology --save
#python3 $ROOT_PROJECT_FOLDER/test.py  --id topology_test --config $ROOT_PROJECT_FOLDER/config_files/config_topology_$SLURM_ARRAY_TASK_ID.json --output-folder topology --save

# PARTIAL AE
#python3 $ROOT_PROJECT_FOLDER/train.py --id partial_ae --config $ROOT_PROJECT_FOLDER/config_files/config_partial_ae.json --output-folder ae_run --save
python3 $ROOT_PROJECT_FOLDER/test.py --id partial_ae --config $ROOT_PROJECT_FOLDER/config_files/config_partial_ae.json --output-folder ae_run --save

# FULL AE
#python3 $ROOT_PROJECT_FOLDER/train.py --id full_ae --config $ROOT_PROJECT_FOLDER/config_files/config_full_ae.json --output-folder ae_run --save
python3 $ROOT_PROJECT_FOLDER/test.py --id full_ae --config $ROOT_PROJECT_FOLDER/config_files/config_full_ae.json --output-folder ae_run --save

conda deactivate
