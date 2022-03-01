from pathlib import Path
import sys
import torch
from torch_geometric.loader import DataLoader
from torch.nn import functional as F
from tqdm import tqdm

FILE_PATH = Path(__file__)
EXPERIMENT_DIR = FILE_PATH.parent
SRC_DIR = FILE_PATH.parents[2] / "src"

sys.path.append(str(SRC_DIR))
from dataset import InjectionMoldingDataset
from modules.fillsimnet import FillSimNet


def loss_function(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cel = F.cross_entropy(input, target)
    class_probs = F.softmax(input, dim=1)
    classes = torch.round(class_probs)
    num_false_negatives = torch.sum(F.relu((target - classes)[:, 0]))
    loss = cel + num_false_negatives * 1000
    return loss


def train(model: torch.nn.Module, train_dataset: InjectionMoldingDataset):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    model = model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = loss_function
    loss_sum = 0
    for d in tqdm(train_loader):
        optimizer.zero_grad()
        d = d.to(device)
        y_out = model(d)
        loss = criterion(y_out, d.y)
        loss_sum += loss.detach().item()
        loss.backward()
        optimizer.step()
    mean_loss = loss_sum / len(train_dataset)
    return model, mean_loss


def main():
    torch.manual_seed(42)
    dataset = InjectionMoldingDataset(0.004, 0.03, skip_processing=True)
    dataset = dataset.shuffle()[:1000]
    train_dataset = dataset[: int(len(dataset) * 0.8)]
    #test_dataset = dataset[int(len(dataset) * 0.8):]

    model = FillSimNet(2, 64, 2, 3)

    for epoch in range(20):
        model, mean_loss = train(model, train_dataset)
        print(f"epoch_loss: {mean_loss}")
        model_path = EXPERIMENT_DIR / f"model_epoch_{epoch}.pt"
        torch.save(model, model_path)


if __name__ == "__main__":
    main()
