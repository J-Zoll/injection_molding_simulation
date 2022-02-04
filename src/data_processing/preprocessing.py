from typing import Iterable, Tuple, List
from scipy.spatial import KDTree

from .util import binary_aggregate, elementwise_distance, discretize, distance


def calculate_fill_states(
        step_size: float,
        fill_times: Iterable[float]
) -> Iterable[Iterable[bool]]:
    """Calculates binary fill_states for a list of continuous node
       fill_times.
    """
    disc_fts = discretize(step_size, fill_times)
    ft_states = binary_aggregate(step_size, disc_fts, condition="smaller_or_equal")
    return ft_states
    

def calculate_edges(
        node_positions: Iterable[Tuple[float, float, float]],
        connection_range: float
) -> Iterable[Tuple[int, int]]:
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


def calculate_elementwise_distances(
        node_positions: List[List[float]],
        edges: Iterable[Tuple[int, int]]
) -> List[List[float]]:
    """Calculates the element-wise distances between connected nodes"""
    distances = []
    for i, j in edges:
        p1 = node_positions[i]
        p2 = node_positions[j]
        distances.append(elementwise_distance(p1, p2))
    return distances


def calculate_distances(
        node_positions: List[Tuple[float, float, float]],
        edges: Iterable[Tuple[int, int]]
) -> List[float]:
    """Calculates the euclidean distances between connected nodes"""
    distances = []
    for i, j in edges:
        p1 = node_positions[i]
        p2 = node_positions[j]
        distances.append(distance(p1, p2))
    return distances


def encode_fill_state(fill_states: Iterable[bool]) -> List[Tuple[float, float]]:
    """Encodes the fill_state as one-hot encoding."""
    enc_filled = (1.0, 0.0)
    enc_not_filled = (0.0, 1.0)

    return [enc_filled if fs else enc_not_filled for fs in fill_states]
