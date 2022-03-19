""" This experiment trains and evaluates a GNN Model with a changed loss function
    The loss function now penalizes false negative node classification. False
    negative node classification represents an "unfilling" of the part which
    does not happen in reality in should therefore be forbidden.

    The model used has the following structure:
        - Encoder: MLP
        - Processor: 3 * GCNConv
        - Decoder: MLP

    The model is trained with a simple Cross Entropy Loss + Penalty for FN classification.
    The model is trained for a maximum of 3 epochs.

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
from dataset import InjectionMoldingDataset
from training import Training
from evaluation import Evaluation
from config import Config


def loss_function(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cel = F.cross_entropy(input, target)
    class_probs = F.softmax(input, dim=1)
    classes = torch.round(class_probs)
    num_false_negatives = torch.sum(F.relu((target - classes)[:, 0]))
    loss = cel + num_false_negatives * 1000
    return loss


if __name__ == "__main__":
    torch.manual_seed(42)

    EXPERIMENT_NAME = "baseline"
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
    dataset = dataset.shuffle()[:2000]

    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    for fold, (train_split, test_split) in enumerate(kf.split(dataset)):
        train_dataset = dataset[train_split]
        test_dataset = dataset[test_split]
        model_to_train = copy.deepcopy(model)

        print(f"\n[Fold {fold}] Training")
        training = Training(
            model_to_train,
            loss_function,
            train_dataset,
            num_epochs=3,
            batch_size=2
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
