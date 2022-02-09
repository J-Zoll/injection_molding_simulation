import os
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch
from torch.nn import DataParallel
import numpy as np
import pandas as pd

from modules.fillsimnet import FillSimNet
from dataset import InjectionMoldingDataset

# data directory
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH_TO_DATASET = os.path.join(os.path.dirname(SCRIPT_PATH), "data")


def split_data(dataset, ratio):
    """Splits a dataset into train and test"""
    split_index = int(len(dataset) * ratio)
    train_data = dataset[:split_index]
    test_data = dataset[split_index:]
    return train_data, test_data


class Training:
    def __init__(
            self,
            model,
            criterion,
            dataset,
            batch_size=8,
            learning_rate=0.01,
            num_epochs=1,
            train_test_data_ratio=0.8
    ):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DataParallel(model)
        self.model.to(device=self.device)
        self.criterion = criterion
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.train_test_data_ratio = train_test_data_ratio

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.train_data, self.test_data = split_data(dataset, train_test_data_ratio)
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_data)

        self.train_losses = []
        self.test_losses = []

    def run(self):
        print("Training...")
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch)
            print(train_loss)
            test_loss = self.test_epoch(epoch)
            print(test_loss)
        print("Done!")

    def train_epoch(self, epoch_id):
        losses = []
        for batch in tqdm(self.train_loader, desc=f"epoch {epoch_id} (train)"):
            batch.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(batch.x, batch.edge_index, batch.edge_weight)
            loss = self.criterion(out, batch.y)
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
        self.train_losses.append(losses)
        return np.mean(losses)

    def test_epoch(self, epoch_id):
        losses = []
        with torch.no_grad():
            for data in tqdm(self.test_loader, desc=f"epoch {epoch_id} (test)"):
                data.to(self.device)
                out = self.model(data.x, data.edge_index, data.edge_weight)
                loss = self.criterion(out, data.y)
                losses.append(loss.item())
        self.test_losses.append(losses)
        return np.mean(losses)

    def get_log(self):
        losses = np.concatenate([self.train_losses, self.test_losses], axis=1)
        epoch_names = [f"epoch_{n}" for n in range(len(losses))]
        df_log = pd.DataFrame(losses.T, columns=epoch_names)
        train_mask = [True for _ in range(len(self.train_losses[0]))]
        test_mask = [False for _ in range(len(self.test_losses[0]))]
        mask = train_mask + test_mask
        df_log["train"] = mask
        return df_log


def main():
    model = FillSimNet(2, 64, 2, 1)
    criterion = torch.nn.CrossEntropyLoss()
    dataset = InjectionMoldingDataset(PATH_TO_DATASET, 0.003, 3, skip_processing=False)

    t = Training(model, criterion, dataset, batch_size=1, num_epochs=1)
    t.run()
    print(t.get_log())


if __name__ == '__main__':
    main()
