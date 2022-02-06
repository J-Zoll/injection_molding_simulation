from typing import Iterable, Tuple, List
from scipy.spatial import KDTree
import torch
from torch import Tensor, LongTensor
import pandas as pd
from torch_geometric.data import Data
import os

from .util import binary_aggregate, elementwise_distance, discretize, distance, parse_to_float_list


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


def process_raw_file(
        raw_file_path: str,
        output_dir: str,
        connection_range: float,
        time_step_size: float
):
    """Process a raw study csv-file into multiple graph data objects stored at output_dir"""
    df_study = pd.read_csv(f"{raw_file_path}.csv")

    raw_dir, raw_file_name = os.path.split(raw_file_path)
    study_name, _ = os.path.splitext(raw_file_name)

    node_positions = [parse_to_float_list(p) for p in df_study.position]

    # calculate edges and edge_distances
    edge_list = calculate_edges(node_positions, connection_range)
    edge_index = LongTensor(edge_list).T
    elementwise_distances = calculate_elementwise_distances(node_positions, edge_list)
    distances = calculate_distances(node_positions, edge_list)

    # calculate fill_states
    fill_states = calculate_fill_states(time_step_size, df_study.fill_time)

    for t, _ in enumerate(fill_states[:-1]):
        # fill_state at the beginning of the time step
        old_fs = Tensor(encode_fill_state(fill_states[t]))
        node_attributes = old_fs

        # (prediction goal) fill_state at the end of the time step
        new_fs = Tensor(encode_fill_state(fill_states[t + 1]))
        target_node_attributes = new_fs

        edge_attributes = Tensor(elementwise_distances)
        edge_weight = Tensor(distances)

        node_positions = Tensor(node_positions)

        # create data object
        data = Data(
            x=node_attributes,
            edge_index=edge_index,
            edge_attr=edge_attributes,
            edge_weight=edge_weight,
            y=target_node_attributes,
            pos=node_positions,
            study=raw_file_path,
            time=t * time_step_size
        )
        data_file_name = f"data_{study_name}_{str(t).zfill(3)}.pt"
        torch.save(data, os.path.join(output_dir, data_file_name))

