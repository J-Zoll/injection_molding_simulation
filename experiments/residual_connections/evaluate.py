import json

import numpy as np
import torch
from torch.nn import functional as F
import sys
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import confusion_matrix

FILE_DIR = Path(__file__).parent
PROJECT_DIR = Path(__file__).parents[2]
SRC_DIR = PROJECT_DIR / "src"
EXP_DIR = PROJECT_DIR / "experiments" / "residual_connections"

sys.path.append(str(SRC_DIR))
from dataset import InjectionMoldingDataset


def evaluate(model, test_dataset):
    conf_mat = np.zeros(shape=(2, 2))
    for d in tqdm(test_dataset):
        out = model(d).detach()
        y_true = d.y[:, 0].numpy()
        y_pred = F.softmax(out, dim=1).round()[:, 0].numpy()
        cm = confusion_matrix(y_true, y_pred)
        conf_mat += cm
    tn, fp, fn, tp = conf_mat.ravel()
    total = np.sum(conf_mat)
    accuracy = (tn + tp) / total
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    ntn, nfp, nfn, ntp = conf_mat.ravel() / total
    return dict(
        total=total,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        ntn=ntn,
        nfp=nfp,
        nfn=nfn,
        ntp=ntp
    )


def main():
    torch.manual_seed(42)

    dataset = InjectionMoldingDataset(0.004, 0.03, skip_processing=True)
    test_dataset = dataset.shuffle()[800:1000]
    model_path = EXP_DIR / "model_epoch_9.pt"
    device = torch.device("cpu")
    model = torch.load(model_path, map_location=device)
    metrics = evaluate(model, test_dataset)
    with open(EXP_DIR / "metrics.json", "w") as json_file:
        json.dump(metrics, json_file)


if __name__ == "__main__":
    main()
