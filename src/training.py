from torch_geometric.loader import DataListLoader
from torch_geometric.nn import DataParallel
from tqdm import tqdm
import torch
import numpy as np
import json


class Training:
    def __init__(
            self,
            model,
            criterion,
            dataset,
            batch_size=1,
            learning_rate=0.01,
            num_epochs=1
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = DataParallel(model)
        self.model.to(self.device)
        self.criterion = criterion
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.train_loader = DataListLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)

        self.train_losses = {}

    def run(self):
        for epoch in range(self.num_epochs):
            mean_epoch_loss = self.train_epoch(epoch)
            print(f"mean_epoch_loss: {mean_epoch_loss}")

    def train_epoch(self, epoch_id):
        epoch_loss = 0
        for data_list in tqdm(self.train_loader, desc=f"epoch_{epoch_id}"):
            self.optimizer.zero_grad()
            output = self.model(data_list)
            y = torch.cat([data.y for data in data_list]).to(output.device)
            loss = self.criterion(output, y)
            loss.backward()
            epoch_loss += loss.detach().item()
            self.optimizer.step()
        mean_epoch_loss = epoch_loss / len(self.train_loader)
        return mean_epoch_loss

    def save_log(self, json_path):
        log = dict(
            epoch_losses=self.train_losses
        )
        with open(json_path, "w") as json_file:
            json.dump(log, json_file)
