from typing import Iterable, Tuple
from data_processing import cleaning
from data_processing.util import binary_aggregate, componentwise_distance, discretize
from torch import Tensor
import os
import pickle
from scipy.spatial import KDTree


def calculate_fill_states(step_size: float, fill_times: Iterable[float]) -> Iterable[Iterable[bool]]:
    """Calculates binary fill_states for a list of continuos node
       fill_times.
    """
    disc_fts = discretize(step_size, fill_times)
    ft_states = binary_aggregate(step_size, disc_fts, condition="smaller_or_equal")
    return ft_states
    

def calculate_edges(node_positions: Iterable[Tuple[float, float, float]], connection_range: float) -> Iterable[Tuple[int, int]]:
    """Calculates edges. Nodes are connected if their distance is
       smaller or equal to connection_range"""
    kd_tree = KDTree(node_positions)

    edges = []
    for i, pos in enumerate(node_positions):
        neighbor_indexes = kd_tree.query_ball_point(
            pos,
            connection_range,
            workers=-1,
            return_sorted=True
        )
        for j in [index for index in neighbor_indexes if index > i]:
            edges.append((i, j))
    return edges


def calculate_distances(node_positions: Iterable[Tuple[float, float, float]], edges: Iterable[Tuple[int, int]] ) -> Iterable[Tuple[float, float, float]]:
    """Calculates the edge-distances for a list of edges."""
    distances = []
    for i, j in edges:
        p1 = node_positions[i]
        p2 = node_positions[j]
        distances.append(componentwise_distance(p1, p2))
    return distances


def encode_fill_state(fill_states: Iterable[bool]) -> Iterable[Tuple[float, float]]:
    """Encodes the fill_state trough one-hot encoding."""
    FILLED = [1.0, 0.0]
    NOT_FILLED = [0.0, 1.0]

    return [FILLED if fs else NOT_FILLED for fs in fill_states]
