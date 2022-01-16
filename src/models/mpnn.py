"""Message Passing Neural Network"""

from torch import Tensor
import torch.nn as nn

from graph import Graph
from models.mlp import MLP
from models.gn import GN, EdgeBlock, NodeBlock, SumAggregator

class MPNN (nn.Module):
    """Message Passing Neural Network Module"""

    def __init__(
        self,
        size_edge_attributes,
        size_node_attributes
        ):
        edge_attribute_update_function = MLP(
            [size_edge_attributes, size_edge_attributes, size_edge_attributes],
            activate_final=False
        )
        node_attribute_update_function = MLP(
            [size_node_attributes, size_node_attributes, size_node_attributes],
            activate_final=False
        )

        edge_block = EdgeBlock(
            edge_attribute_update_function,
            use_edge_attributes=True,
            use_sender_node_attributes=True,
            use_receiver_node_attributes=True,
            use_global_attributes=False
        )
        node_block = NodeBlock(
            node_attribute_update_function,
            receiving_edge_aggregator=SumAggregator(),
            use_node_attributes=True,
            use_receiving_edge_attributes=True,
            use_sending_edge_attributes=False,
            use_global_attributes=False
        )

        self.gn_model = GN(
            edge_block=edge_block,
            node_block=node_block
        )

        super().__init__()


    def forward(self, graph: Graph):
        return self.gn_model(graph)
