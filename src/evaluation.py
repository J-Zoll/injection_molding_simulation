import json
import multiprocessing as mp

from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, balanced_accuracy_score,\
    f1_score, roc_auc_score
import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data


class Evaluation:
    """Evaluates a module and captures various scores"""

    def __init__(self, model: torch.nn.Module, dataset: Dataset):
        self.model = model
        self.dataset = dataset

        self.tn = None
        self.fp = None
        self.fn = None
        self.tp = None
        self.total = None
        self.accuracy = None
        self.balanced_accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None
        self.roc_auc = None

        self._run()

    def _run(self):
        """Make predictions and calculate scores"""
        y_true, y_pred = self.make_predictions()
        self.tn, self.fp, self.fn, self.tp = confusion_matrix(y_true, y_pred).ravel()
        self.total = len(y_pred)
        self.accuracy = accuracy_score(y_true, y_pred)
        self.balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        self.precision = precision_score(y_true, y_pred)
        self.recall = recall_score(y_true, y_pred)
        self.f1 = f1_score(y_true, y_pred)
        self.roc_auc = roc_auc_score(y_true, y_pred)

    def make_predictions(self):
        """Get truth and predictions from dataset"""
        y_true = []
        y_pred = []
        with mp.Pool() as pool:
            ys = [x for x in tqdm(pool.imap_unordered(self.predict, self.dataset), total=len(self.dataset))]
        for yt, yp in ys:
            y_true += yt
            y_pred += yp
        return y_true, y_pred

    def predict(self, d: Data):
        with torch.no_grad():
            y_true = d.x[:, 0].tolist()
            pred_logits = self.model(d.x, d.edge_index, d.edge_weight)
            pred_prob = F.softmax(pred_logits, dim=1)
            y_pred = pred_prob.round()[:, 0].tolist()
        return y_true, y_pred

    def save_scores(self, json_file_path):
        """Save scores to a json file"""
        scores = dict(
            tn=self.tn,
            fp=self.fp,
            fn=self.fn,
            tp=self.tp,
            total=self.total,
            accuracy=self.accuracy,
            balanced_accuracy=self.balanced_accuracy,
            precision=self.precision,
            recall=self.recall,
            f1=self.f1,
            roc_auc=self.roc_auc
        )
        with open(json_file_path, "w") as json_file:
            json.dump(scores, json_file)
