""" This experiment trains and evaluates a baseline GNN Model

    The model used has the following structure:
        - Encoder: MLP
        - Processor: 3 * GCNConv
        - Decoder: MLP

    The model is trained with a simple Cross Entropy Loss.
    The model is trained for a maximum of 5 epochs.

    The dataset is generated based on the following parameters:
        - connection_range: 0.04
        - time_step_size: 0.003

    The model is evaluated using 5-fold cross validation.
"""
import copy
import json
import os
import numpy as np

import torch
import torch.nn.functional as F
import sys
from pathlib import Path
from sklearn.model_selection import KFold

SRC_DIR = Path(__file__).parents[2] / "src"
sys.path.append(str(SRC_DIR))

from modules.fillsimnet import FillSimNet
from modules.mlp import MLP
from dataset import InjectionMoldingDataset
from training import Training
from evaluation import Evaluation
from config import Config

if __name__ == "__main__":
    torch.manual_seed(42)

    EXPERIMENT_NAME = "baseline_gnn_model"
    CURRENT_EXPERIMENT_DIR = Config.EXPERIMENT_DIR / EXPERIMENT_NAME
    MODEL_OUTPUT_DIR = CURRENT_EXPERIMENT_DIR / "models"
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCORE_OUTPUT_DIR = CURRENT_EXPERIMENT_DIR / "scores"
    SCORE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # hyper parameters
    CONNECTION_RANGE = 0.04
    TIME_STEP_SIZE = 0.003
    NUM_GNN_LAYERS = 3

    # get dataset
    dataset = InjectionMoldingDataset(CONNECTION_RANGE, TIME_STEP_SIZE, skip_processing=True)

    # get model
    model = FillSimNet(2, 128, 2, NUM_GNN_LAYERS)

    # train
    K = 5
    kf = KFold(n_splits=K)
    dataset = dataset.shuffle()
    for fold, (train_split, test_split) in enumerate(kf.split(dataset)):
        train_dataset = dataset[train_split]
        test_dataset = dataset[test_split]
        model_to_train = copy.deepcopy(model)

        print(f"\n[Fold {fold}] Training")
        training = Training(
            model_to_train,
            F.cross_entropy,
            train_dataset,
            num_epochs=5,
            batch_size=4
        )
        training.run()
        training.save_log(MODEL_OUTPUT_DIR / f"log_fold_{fold}.json")
        trained_model = training.model

        # save model
        model_file_path = MODEL_OUTPUT_DIR / f"model_fold_{fold}.pt"
        torch.save(trained_model, model_file_path)

        # evaluate model
        print(f"\n[Fold {fold}] Evaluation")
        evaluation = Evaluation(
            trained_model,
            test_dataset
        )
        evaluation.run()
        # save score
        score_file_path = SCORE_OUTPUT_DIR / f"score_fold_{fold}.json"
        evaluation.save_scores(score_file_path)


    # train model on whole data
    print("\n[Whole Data]: Training")
    model_to_train = copy.deepcopy(model)
    training = Training(
        model_to_train,
        F.cross_entropy,
        dataset,
        num_epochs=3,
        batch_size=4
    )
    training.run()
    training.save_log(MODEL_OUTPUT_DIR / f"log_whole_data.json")
    trained_model = training.model

    # save model
    model_file_path = MODEL_OUTPUT_DIR / f"model_whole_data.pt"
    torch.save(trained_model, model_file_path)

    # take mean of every score as a combined score of the experiment
    keys = []
    values = []
    for fold_score in os.listdir(SCORE_OUTPUT_DIR):
        with open(SCORE_OUTPUT_DIR / fold_score, "r") as json_file:
            scores = json.load(json_file)
            keys = scores.keys()
            values.append(list(scores.values()))
    mean_values = [np.mean(vs) for vs in np.array(values).T]
    scores_whole_data = dict(zip(keys, mean_values))

    # save scores for whole data
    json_file_path = SCORE_OUTPUT_DIR / "score_whole_data.json"
    with open(json_file_path, "w") as json_file:
        json.dump(scores_whole_data, json_file)
