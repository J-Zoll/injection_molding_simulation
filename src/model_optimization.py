from typing import Dict, List
import os
import os.path as osp
import torch
import pandas as pd
import json
import time

DEBUG = True
HYPER_PARAMETERS = {
    "connection_range": [0.003, 0.004, 0.005, 0.006],
    "time_step_size": [0.02, 0.05, 0.08, 0.12],
    "num_conv_layers": [1, 2, 3, 4, 5, 6]
}
OUTPUT_DIR = "/Users/jonas/Documents/Bachelorarbeit/injection_molding_simulation/data/training"
TRAINED_MODEL_DIR = osp.join(OUTPUT_DIR, "trained_models")
TRAINING_LOG_DIR = osp.join(OUTPUT_DIR, "training_logs")
TRAINING_CONFIGURATION_DIR = osp.join(OUTPUT_DIR, "training_configurations")


def main():
    all_hyperparameter_combinations = get_combinations(HYPER_PARAMETERS)

    for hyper_parameters in all_hyperparameter_combinations:
        trained_model, training_log = run_training(hyper_parameters)
        save_training_data(trained_model, training_log)


def get_combinations(parameters: Dict) -> List[Dict]:
    """Returns all parameter combinations"""
    param_key, param_values = list(parameters.items())[0]
    other_parameters = dict(list(parameters.items())[1:])

    if other_parameters == {}:
        return [{param_key: pv} for pv in param_values]

    else:
        other_param_combinations = get_combinations(other_parameters)
        combinations = []
        for pv in param_values:
            for opc in other_param_combinations:
                comb = opc.copy()
                comb[param_key] = pv
                combinations.append(comb)
        return combinations


def run_training(hyperparameters):
    pass


def save_training_data(hyperparameters, trained_model, training_log: pd.DataFrame):
    make_output_dirs()
    key = get_key()

    # save trained model
    path_trained_model = get_path_trained_model(key)
    torch.save(trained_model, path_trained_model)

    # save training log
    path_training_log = get_path_training_log(key)
    training_log.to_csv(path_training_log, index=False)

    # save training configuration
    training_configuration = {
        "hyperparameters": hyperparameters,
        "trained_model": path_trained_model,
        "training_log": path_training_log
    }
    path_training_configuration = get_path_training_configuration(key)
    with open(path_training_configuration, "w") as json_file:
        json.dump(training_configuration, json_file)


def get_path_trained_model(key: str) -> str:
    file_name = f"trained_model_{key}.pickle"
    return osp.join(TRAINED_MODEL_DIR, file_name)


def get_path_training_log(key: str) -> str:
    file_name = f"training_log_{key}.csv"
    return osp.join(TRAINING_LOG_DIR, file_name)


def get_path_training_configuration(key: str) -> str:
    file_name = f"training_configuration_{key}.json"
    return osp.join(TRAINING_CONFIGURATION_DIR, file_name)


def get_key(*args, **kwargs) -> str:
    return str(time.time_ns())


def make_output_dirs():
    # root dir
    if not osp.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    # dir for trained models
    if not osp.isdir(TRAINED_MODEL_DIR):
        os.mkdir(TRAINED_MODEL_DIR)
    # dir for training logs
    if not osp.isdir(TRAINING_LOG_DIR):
        os.mkdir(TRAINING_LOG_DIR)
    # dir for training configurations
    if not osp.isdir(TRAINING_CONFIGURATION_DIR):
        os.mkdir(TRAINING_CONFIGURATION_DIR)


if __name__ == "main":
    main()
