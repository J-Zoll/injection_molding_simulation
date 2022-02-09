from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from .mlp import MLP


class FillSimNet (nn.Module):
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

    def forward(self, x, edge_index, edge_weight):
        # encoding
        x = self.encoder(x)

        # message passing
        for layer in self.processor:
            x = layer(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)

        # decoding
        x = self.decoder(x)

        # classification
        # x = F.softmax(x, dim=1)

        return x
