from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from mlp import MLP


class FillSimNet (nn.Module):
    def __init__(self, node_inp_size, node_emb_size, node_out_size) -> None:
        super().__init__()

        # encoder
        self.encoder = MLP(node_inp_size, node_emb_size, node_emb_size)

        # processor
        self.conv1 = GCNConv(node_emb_size, node_emb_size)
        self.conv2 = GCNConv(node_emb_size, node_emb_size)

        # decoder
        self.decoder = MLP(node_emb_size, node_emb_size, node_out_size)

    def forward(self, x, edge_index, edge_weight):
        # encoding
        x = self.encoder(x)

        # message passing
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu()

        # decoding
        x = self.decoder(x)

        # classification
        out = F.softmax(x, dim=1)

        return out
