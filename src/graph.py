"""A Data Object representing a Graph"""

from torch import Tensor
import torch
from torch._C import BoolTensor

class Graph:
    """ Data Object representing a Graph"""

    def __init__(
        self,
        node_attribues: Tensor,
        edge_attributes: Tensor,
        global_attributes: Tensor,
        edges: Tensor
    ):
        """Constructs a Graph
        Args:
        node_attributes:
            Attributes of the nodes of the graph.
            <<type>>: [num_nodes, size_node_attributes]-Tensor
        edge_attributes:
            Attributes of the edges of the graph.
            <<type>>: [num_edges, size_edge_attributes]-Tensor
        global_attributes:
            Global attributes of the graph.
            <<type>>: [size_global_atributes]-Tensor
        edges:
            Connectivity of the graph. An edge from node i to node j is encoded as an entry
            Tensor([i, j]) in edges.
            <<type>>: [num_edges, 2]-Tensor
        """
        self.node_attributes = node_attribues
        self.edge_attributes = edge_attributes
        self.global_attributes = global_attributes
        self.edges = edges
        self.nodes = Tensor(list(range(len(node_attribues))))

    def get_sender_attributes(self) -> Tensor:
        """Returns the attributes of all sender nodes"""
        sender_indexes = self.edges[:, 0]
        sender_attributes = sender_indexes.clone().apply_(lambda i: self.node_attributes[i])
        return sender_attributes


    def get_receiver_attributes(self) -> Tensor:
        """Returns the attributes of the all receiver nodes"""
        receiver_indexes = self.edges[:, 1]
        receiver_attributes = receiver_indexes.clone().apply_(lambda i: self.node_attributes[i])
        return receiver_attributes


    def get_edge_attributes_ingoing(self, node: int) -> Tensor:
        """Returns the attributes of all edges going in a specific node of the graph"""
        mask_ingoing_edge = BoolTensor([node == r for s, r in self.edges]).reshape([-1, 1])
        size_edge_attributes = self.edge_attributes.shape[1]
        ingoing_edge_attributes = torch.masked_select(self.edge_attributes, mask_ingoing_edge).reshape([-1, size_edge_attributes])
        return ingoing_edge_attributes.clone()



    def get_edge_attributes_outgoing(self, node: int) -> Tensor:
        """Returns the attributes of all edges going out of a specific node of the graph"""
        mask_outgoing_edge = BoolTensor([node == s for s, r in self.edges]).reshape([-1, 1])
        size_edge_attributes = self.edge_attributes.shape[1]
        outgoing_edge_attributes = torch.masked_select(self.edge_attributes, mask_outgoing_edge).reshape([-1, size_edge_attributes])
        return outgoing_edge_attributes.clone()
