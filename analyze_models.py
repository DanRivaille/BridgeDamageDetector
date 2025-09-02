import os
import json

import pandas as pd
import numpy as np

from src.damage_detector.CommonPath import CommonPath

BASE_MODELS_STEP = "base_models"
FINE_TUNED_MODELS_STEP = "fine-tuned_models"
PRUNED_MODELS_STEP = "pruned_models"

base_directories = {BASE_MODELS_STEP, FINE_TUNED_MODELS_STEP, PRUNED_MODELS_STEP, "first_experiments", "proportion_experiments"}


def load_json_data(json_pathfile: str) -> dict:
    with open(json_pathfile, "r") as json_file:
        json_data = json.load(json_file)

    return json_data


def get_metrics(model_folder: str, step_name: str, technique: str, proportion: int) -> []:
    base_pathfile = os.path.join(model_folder, step_name)
    #if not os.path.isdir(base_pathfile):
        #return []

    metric_per_model = []
    for model_directory in os.listdir(base_pathfile):
        model_directory_full_path = os.path.join(base_pathfile, model_directory)

        metrics_pathfile = os.path.join(model_directory_full_path, "metrics.json")

        #if not os.path.isfile(metrics_pathfile):
            #continue

        metric_raw_data = load_json_data(metrics_pathfile)

        metric_data = (
            model_directory,
            technique,
            proportion,
            step_name,
            metric_raw_data['validation_loss'],
            metric_raw_data['memory_before_inference'],
            metric_raw_data['inference_time'],
            metric_raw_data['test_results']['max_f1'],
            metric_raw_data['test_results']['max_auc'],
        )

        metric_per_model.append(metric_data)

    return metric_per_model


def extract_directory_info(directory: str) -> tuple:
    tokens = directory.split("_")
    proportion = int(tokens[1].split("-")[0])
    technique = tokens[0]
    return technique, proportion


def main():
    directories = os.listdir(CommonPath.MODEL_PARAMETERS_FOLDER.value)

    metrics = []
    for directory in directories:
        directory_full_path = os.path.join(CommonPath.MODEL_PARAMETERS_FOLDER.value, directory)

        if os.path.isdir(directory_full_path) and directory not in base_directories:
            technique, proportion = extract_directory_info(directory)
            current_run_metric = get_metrics(directory_full_path, BASE_MODELS_STEP, technique, proportion)
            metrics = metrics + current_run_metric

            current_run_metric = get_metrics(directory_full_path, PRUNED_MODELS_STEP, technique, proportion)
            metrics = metrics + current_run_metric

            current_run_metric = get_metrics(directory_full_path, FINE_TUNED_MODELS_STEP, technique, proportion)
            metrics = metrics + current_run_metric

    metrics_df = pd.DataFrame(np.array(metrics), columns=["ModelName", "Technique", "Proportion", "Step",
                                                          "validation_loss", "memory_used", "inference_time",
                                                          "f1_score", "auc_score"])

    base_path = '/home/ivan.santos/repositories/BridgeDamageDetector/sbatch_scripts'
    metrics_df.to_csv(os.path.join(base_path, "models_metric.csv"), index=False)


if __name__ == '__main__':
    main()
