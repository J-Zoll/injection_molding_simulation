import torch
import sys
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from multiprocessing import Pool
import numpy as np

SRC_DIR = Path(__file__).parents[2] / "src"
sys.path.append(str(SRC_DIR))

from dataset import InjectionMoldingDataset
from config import Config


def unchanged_metrics(d):
    y_true = d.y[:, 0]
    y_pred = torch.randint(low=0, high=2, size=y_true.size())
    #y_pred = d.x[:, 0]

    return calculate_metrics(y_true, y_pred)


def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel().tolist()
    total = len(y_pred)

    return [tn, fp, fn, tp, total]


if __name__ == "__main__":
    torch.manual_seed(42)

    EXPERIMENT_NAME = "unchanged"
    CURRENT_EXPERIMENT_DIR = Config.EXPERIMENT_DIR / EXPERIMENT_NAME
    SCORE_OUTPUT_DIR = CURRENT_EXPERIMENT_DIR / "scores"
    SCORE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # hyper parameters
    CONNECTION_RANGE = 0.04
    TIME_STEP_SIZE = 0.003
    NUM_GNN_LAYERS = 3

    # get dataset
    dataset = InjectionMoldingDataset(CONNECTION_RANGE, TIME_STEP_SIZE, skip_processing=True)
    dataset = dataset.shuffle()[:2000]

    pool = Pool()
    scores = [_ for _ in tqdm(pool.imap_unordered(unchanged_metrics, dataset), total=len(dataset))]
    scores = np.array(scores)

    acc_scores = [np.sum(s) for s in scores.T]
    tn, fp, fn, tp, total = acc_scores

    accuracy = (tn + tp) / total
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print(f"tn: {tn}")
    print(f"fp: {fp}")
    print(f"fn: {fn}")
    print(f"tp: {tp}")
    print(f"total: {total}")
    print(f"accuracy: {accuracy}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")
