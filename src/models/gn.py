"""Graph Network Module"""

from abc import ABC, abstractmethod

import torch
from torch import nn
from torch import Tensor

from graph import Graph


class Aggregator (nn.Module, ABC):
    """Module to perform an aggregation operation on a Tensor"""

    @abstractmethod
    def forward(self, to_aggregate: Tensor) -> Tensor:
        """Defines the computation performed at every call.
        Args:
        to_aggregate:
            Tensor to aggregate at axis 0.
            <<type>>: Tensor
        """


class SumAggregator (Aggregator):
    """Aggregator which aggregates a Tensor via summation"""

    def forward(self, to_aggregate: Tensor):
        return torch.sum(to_aggregate, axis=0)


class EdgeBlock (nn.Module):
    """Module representing a update of a graphs edge attributes"""
    def __init__(
        self,
        edge_attribute_update_function,
        use_edge_attributes=True,
        use_sender_node_attributes=True,
        use_receiver_node_attributes=True,
        use_global_attributes=True
    ):
        """Constructs an Edge Block
        Args:
        edge_attribute_update_function:
            Function to calculate new edge attributes
            Kwargs:
                - edge_attributes: [<num_edges>, <size_edge_attributes>]-Tensor, optional
                - sender_node_attributes: [<size_node_attributes>]-Tensor, optional
                - receiver_node_attributes: [<size_node_attributes>]-Tensor, optional
                - global_attributes: [<size_global_attributes>]-Tensor, optional
            Returns:
                - new_edge_attributes: [<num_edges>, <size_edge_attributes>]-Tensor
            <<type>>: (**kwargs) -> [num_edges, size_edge_attributes]-Tensor
        use_edge_attributes:
            Whether or not to pass the edge_attributes to the edge_attribute_update_function.
            <<type>>: bool
            <<default>>: True
        use_sender_node_attributes:
            Whether or not to pass the sender_node_attributes to the
            edge_attribute_update_function.
            <<type>>: bool
            <<default>>: True
        use_receiver_node_attributes:
            Whether or not to pass the receiver_node_attributes to the
            edge_attribute_update_function.
            <<type>>: bool
            <<default>>: True
        use_global_attributes:
            Whether or not to pass the global_attributes to the node_attribute_update_function.
            <<type>>: bool
            <<default>>: True
        """
        super().__init__()

        self._edge_attribute_update_function = edge_attribute_update_function
        self._use_edge_attributes = use_edge_attributes
        self._use_sender_node_attributes = use_sender_node_attributes
        self._use_receiver_node_attributes = use_receiver_node_attributes
        self._use_global_attributes = use_global_attributes


    def forward(self, graph: Graph):
        """Defines the computation performed at every call.
        Args:
        graph:
            The graph in wich the edge_attributes get updated
            <<type>>: Graph
        """
        kwargs = {}

        if self._use_edge_attributes:
            kwargs["edge_attributes"] = graph.edge_attributes
        if self._use_sender_node_attributes:
            kwargs["sender_node_attributes"] = graph.get_sender_attributes()
        if self._use_receiver_node_attributes:
            kwargs["receiver_node_attributes"] = graph.get_receiver_attributes()
        if self._use_global_attributes:
            kwargs["global_attributes"] = graph.global_attributes

        new_edge_attributes = self._edge_attribute_update_function(**kwargs)
        graph.edge_attributes = new_edge_attributes

        return graph


class NodeBlock (nn.Module):
    """Module representing a update of a graphs node attributes"""

    def __init__(
        self,
        node_attribute_update_function,
        receiving_edge_aggregator: Aggregator = None,
        sending_edge_aggregator: Aggregator = None,
        use_receiving_edge_attributes=True,
        use_sending_edge_attributes=True,
        use_node_attributes=True,
        use_global_attributes=True
    ):
        """Constructs an Node Block
        Args:
        receiving_edge_aggregator:
            Aggregator to aggregate the edge_attributes of incoming edges to a node
            <<type>>: Aggregator
        sending_edge_aggregator:
            Aggregator to aggregate the edge_attributes of outgoing edges away from a node
            <<type>>: Aggregator
        node_attribute_update_function:
            Function to calculate new node attributes
            Kwargs:
                - receiving_edge_attributes: [size_edge_attributes]-Tensor
                - sending_edge_attributes: [size_edge_attributes]-Tensor
                - node_attributes: [num_nodes, size_node_attributes]-Tensor
                - global_attributes: [size_global_attributes]-Tensor
            Returns:
                - new_node_attributes: [<num_nodes>, <size_node_attributes>]-Tensor
            <<type>>: (**kwargs) -> [num_nodes, size_node_attributes]-Tensor
        use_receiving_edge_attributes:
            Whether or not to pass the receiving_edge_attributes to the
            node_attribute_update_function.
            <<type>>: bool
            <<default>>: True
        use_sending_edge_attributes:
            Whether or not to pass the sending_edge_attributes to the
            node_attribute_update_function.
            <<type>>: bool
            <<default>>: True
        use_node_attributes:
            Whether or not to pass the node_attributes to the
            node_attribute_update_function.
            <<type>>: bool
            <<default>>: True
        use_global_attributes:
            Whether or not to pass the global_attributes to the
            node_attribute_update_function.
            <<type>>: bool
            <<default>>: True
        """
        super().__init__()
        self._receiving_edge_aggregator = receiving_edge_aggregator
        self._sending_edge_aggregator = sending_edge_aggregator
        self._node_attribute_update_function = node_attribute_update_function
        self._use_receiving_edge_attributes = use_receiving_edge_attributes
        self._use_sending_edge_attributes = use_sending_edge_attributes
        self._use_node_attributes = use_node_attributes
        self._use_global_attributes = use_global_attributes


    def forward(self, graph:Graph):
        """Defines the computation performed at every call.
        Args:
        graph:
            The graph in wich the node_attributes get updated
            <<type>>: Graph
        """
        kwargs = {}

        if self._use_receiving_edge_attributes:
            rec_edge_attr_list = []
            for node in graph.nodes:
                edge_attrs_to_node = graph.get_edge_attributes_ingoing(node)
                aggr_edge_attrs_to_node = self._receiving_edge_aggregator(edge_attrs_to_node)
                rec_edge_attr_list.append(aggr_edge_attrs_to_node)
            rec_edge_attrs = torch.cat(rec_edge_attr_list)
            kwargs['receiving_edge_attributes'] = rec_edge_attrs

        if self._use_sending_edge_attributes:
            sen_edge_attr_list = []
            for node in graph.nodes:
                edge_attrs_from_node = graph.get_edge_attributes_outgoing(node)
                aggr_edge_attrs_from_node = self._sending_edge_aggregator(edge_attrs_from_node)
                rec_edge_attr_list.append(aggr_edge_attrs_from_node)
            sen_edge_attrs = torch.cat(sen_edge_attr_list)
            kwargs['sending_edge_attributes'] = sen_edge_attrs

        if self._use_node_attributes:
            kwargs['node_attributes'] = graph.node_attributes

        if self._use_global_attributes:
            kwargs['global_attributes'] = graph.global_attributes

        new_node_attributes = self._node_attribute_update_function(**kwargs)
        graph.node_attributes = new_node_attributes

        return graph


class GlobalBlock (nn.Module):
    """Module representing a update of a graphs global attributes"""

    def __init__(
        self,
        global_attributes_update_function,
        edge_attribute_aggregator: Aggregator,
        node_attribute_aggregator: Aggregator,
        use_edge_attributes: bool = True,
        use_node_attributes: bool = True,
        use_global_attributes: bool = True,
    ):
        """Constructs a global block
        Args:
        global_attribute_update_function:
            Function to calculate new global attributes
            Kwargs:
                - edge_attributes: [size_edge_attributes]-Tensor
                - node_attributes: [size_node_attributes]-Tensor
                - global_attributes: [size_global_attributes]-Tensor
            Returns:
                - new_global_attributes: [<size_global_attributes>]-Tensor
            <<type>>: (**kwargs) -> [size_global_attributes]-Tensor
        edge_attribute_aggregator:
            Aggregator to aggregate the attributes of all edges of the graph.
            <<type>>: Aggregator
        node_attribute_aggregator:
            Aggregator to aggregate the attributes of all nodes of the graph.
            <<type>>: Aggregator
        use_edge_attributes:
            Whether or not to pass the edge_attributes to the
            global_attribute_update_function
            <<type>>: bool
            <<default>>: True
        use_node_attributes:
            Whether or not to pass the node_attributes to the
            global_attribute_update_function
            <<type>>: bool
            <<default>>: True
        use_global_attributes:
            Whether or not to pass the global_attributes to the
            global_attribute_update_function
            <<type>>: bool
            <<default>>: True
        """
        super().__init__()

        self._global_attribute_update_function = global_attributes_update_function
        self._edge_attribute_aggregator = edge_attribute_aggregator
        self._node_attribute_aggregator = node_attribute_aggregator
        self._use_edge_attributes = use_edge_attributes
        self._use_node_attributes = use_node_attributes
        self._use_global_attributes = use_global_attributes


    def forward(self, graph: Graph):
        """Defines the computation performed at every call.
        Args:
        - graph:
            The graph in wich the global_attributes get updated
            <<type>>: Graph
        Returns:
        - updated_graph:
            The input graph with updated global attributes
            <<type>>: Graph
        """
        kwargs = {}

        if self._use_edge_attributes:
            aggr_edge_attrs = self._edge_attribute_aggregator(graph.edge_attributes)
            kwargs["edge_attributes"] = aggr_edge_attrs

        if self._use_node_attributes:
            aggr_node_attrs = self._node_attribute_aggregator(graph.node_attributes)
            kwargs["node_attributes"] = aggr_node_attrs

        if self._use_global_attributes:
            kwargs["global_attributes"] = graph.global_attributes

        new_global_attributes = self._global_attribute_update_function(**kwargs)
        graph.global_attributes = new_global_attributes

        return graph


class GN (nn.Module):
    """Module representing a Graph Network"""
    def __init__(
        self,
        edge_block = None,
        node_block = None,
        global_block = None
    ):
        super().__init__()

        blocks = []
        if edge_block:
            blocks.append(edge_block)
        if node_block:
            blocks.append(node_block)
        if global_block:
            blocks.append(global_block)

        self.block_sequence = nn.Sequential(blocks)

    def forward(self, graph: Graph):
        """Defines the computation performed at every call.
        Args:
        - graph:
            <<type>>: Graph
        Returns:
        - updated_graph:
            <<type>>: Graph
        """
        return self.block_sequence(graph)
