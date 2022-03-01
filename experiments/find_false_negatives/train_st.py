import torch
from torch.nn import functional as F
import sys
from tqdm import tqdm
from pathlib import Path
from torch_geometric.loader import DataLoader
from torch.optim import Adam

FILE_DIR = Path(__file__).parent
PROJECT_DIR = Path(__file__).parents[2]
SRC_DIR = PROJECT_DIR / "src"

sys.path.append(str(SRC_DIR))
from dataset import InjectionMoldingDataset
from modules.fillsimnet import FillSimNet


def get_num_fn(input, target):
    classes = F.softmax(input, dim=1).round()
    num_false_negatives = torch.sum(F.relu((target - classes)[:, 0]))
    return num_false_negatives


def loss_function(input, target):
    cel = F.cross_entropy(input, target)
    num_fn = get_num_fn(input, target)
    return cel + num_fn


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def train_epoch(model, data_loader, criterion, device):
    optimizer = Adam(params=model.parameters(), lr=0.0005)
    model = model.to(device)
    loss_sum = 0
    for batch in tqdm(data_loader):
        optimizer.zero_grad()
        batch.to(device)
        out = model(batch)
        loss = criterion(out, batch.y)
        loss_sum += loss.detach().item()
        loss.backward()
        optimizer.step()
    mean_loss = loss_sum / len(data_loader)
    return model, mean_loss


def main():
    torch.manual_seed(42)

    dataset = InjectionMoldingDataset(0.004, 0.03, skip_processing=True)
    train_dataset = dataset.shuffle()[:800]
    train_loader = DataLoader(train_dataset, batch_size=10, num_workers=1)
    model = FillSimNet(2, 64, 2, 3)
    criterion = loss_function
    device = get_device()

    for epoch in range(10):
        model, loss = train_epoch(model, train_loader, criterion, device)
        print(f"epoch_{epoch}_loss: {loss}\n")
        torch.save(model, FILE_DIR / f"model_epoch_{epoch}.pt")
        with open(FILE_DIR / "losses.txt", "a") as loss_file:
            loss_file.write(f"epoch_{epoch}: {loss}\n")


if __name__ == "__main__":
    main()
