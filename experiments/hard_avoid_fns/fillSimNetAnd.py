from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import sys
import torch
from pathlib import Path

PROJECT_DIR = Path(__file__).parents[2]
SRC_DIR = PROJECT_DIR / "src"
sys.path.append(str(SRC_DIR))

from modules.mlp import MLP


class FillSimNetAnd (nn.Module):
    def __init__(self, node_inp_size, node_emb_size, node_out_size, num_mp_layers) -> None:
        super().__init__()

        # encoder
        self.encoder = MLP(node_inp_size, node_emb_size, node_emb_size)

        # processor
        mp_layers = []
        for _ in range(num_mp_layers):
            mp_layers.append(GCNConv(node_emb_size, node_emb_size))
        self.processor = nn.ModuleList(mp_layers)

        # decoder
        self.decoder = MLP(node_emb_size, node_emb_size, node_out_size)

        self.weight_constant = torch.tensor([2, 0])

    def forward(self, data):
        # encoding
        x = self.encoder(data.x)

        # message passing
        for layer in self.processor:
            x = layer(x, data.edge_index, edge_weight=data.edge_weight)
            x = F.relu(x)

        # decoding
        x = self.decoder(x)

        x = F.softmax(x, dim=1)
        x = x + (data.x * self.weight_constant.to(x.device))

        return x
