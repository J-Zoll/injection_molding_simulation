import torch
from torch import nn
from torch_geometric.nn import GCNConv


class SimNet(torch.nn.Module):
    def __init__(self, num_mp_layers):
        super().__init__()
        hidden_size = 64
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.processor = nn.ModuleList([GCNConv(hidden_size, hidden_size) for _ in range(num_mp_layers)])
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.ReLU()
        )

    def forward(self, data):
        x = data.x

        x = self.encoder(x)

        for layer in self.processor:
            x = layer(x, edge_index=data.edge_index, edge_weight=data.edge_weight)

        out = self.decoder(x)
        out = torch.sigmoid(out)

        return out
