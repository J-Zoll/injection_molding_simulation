import os
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch
import pandas as pd

from modules.fillsimnet import FillSimNet
from dataset import InjectionMoldingDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(loader: DataLoader, optimizer, criterion, model):
    running_loss = 0
    correct_nodes = 0
    total_nodes = 0
    accuracy = 0

    for batch in tqdm(loader, desc="train"):
        batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_weight)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct, total = evaluate(out, batch.y)
        correct_nodes += correct
        total_nodes += total
        accuracy = correct_nodes / total_nodes

    return running_loss, accuracy


def evaluate(prediction, ground_truth):
    pred_classes = prediction.argmax(axis=1)
    true_classes = ground_truth.argmax(axis=1)
    correct_nodes = torch.sum(pred_classes == true_classes).item()
    total_nodes = len(prediction)

    return correct_nodes, total_nodes


def test_epoch(loader: DataLoader, model):
    correct_nodes = 0
    total_nodes = 0
    with torch.no_grad():
        for data in tqdm(loader, desc="test"):
            data.to(device)
            pred = model(data.x, data.edge_index, data.edge_weight)
            correct, total = evaluate(pred, data.y)
            correct_nodes += correct
            total_nodes += total
    return correct_nodes / total_nodes


def main():
    torch.manual_seed(42)

    # model_training parameters
    BATCH_SIZE = 8
    LEARNING_RATE = 0.01
    NODE_EMBEDDING_SIZE = 128
    NUM_EPOCHS = 1000
    CONNECTION_RANGE = 0.005
    TIME_STEP_SIZE = .05

    # data directory
    SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
    PATH_TO_DATASET = os.path.join(os.path.dirname(SCRIPT_PATH), "data")

    dataset = InjectionMoldingDataset(PATH_TO_DATASET, CONNECTION_RANGE, TIME_STEP_SIZE, skip_processing=True)

    # split data in training data and test data
    split_index = int(len(dataset) * 0.8)
    train_data = dataset[:split_index]
    test_data = dataset[split_index:]

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data)

    model = FillSimNet(dataset.num_features, NODE_EMBEDDING_SIZE, 2)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting model_training")

    stats = []
    for epoch in range(NUM_EPOCHS):
        loss, train_accuracy = train_epoch(train_loader, optimizer, criterion, model)
        torch.save(model.state_dict(), "./data/trained_models/trained_model.pickle")
        test_accuracy = test_epoch(test_loader, model)
        stats.append((epoch, loss, train_accuracy, test_accuracy))
        print(f"Epoch {epoch} | loss: {loss}    train_accuracy: {train_accuracy}    test_accuracy: {test_accuracy}")

        # save model_training statistics
        columns = ("epoch", "loss", "train_accuracy", "test_accuracy")
        df_stats = pd.DataFrame(stats, columns=columns)
        path_to_csv = "training_log.csv"
        df_stats.to_csv(path_to_csv, index=False)

    print("Finished model_training")


if __name__ == '__main__':
    main()
