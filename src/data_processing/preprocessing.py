from scipy.spatial import KDTree
import torch
import pandas as pd
from torch_geometric.data import Data
import os.path as osp
import numpy as np


def parse_to_float_list(string: str):
    string = string[1:-1]  # remove parenthesis
    str_numbers = string.split(", ")
    numbers = [float(f) for f in str_numbers]
    return tuple(numbers)


def get_fill_states(fill_times: np.ndarray, step_size: float) -> np.ndarray:
    """Calculates binary fill_states for a list of continuous node
       fill_times.
    """
    num_steps = np.max(fill_times) // step_size + 1
    ts = np.arange(num_steps + 1) * step_size
    fill_states = np.array([fill_times <= t for t in ts])
    return fill_states
    

def get_edges(pos: np.ndarray, connection_range: float) -> np.ndarray:
    """Calculates edges. Nodes are connected if their distance is
       smaller or equal to connection_range"""
    kd_tree = KDTree(pos)
    edges = []
    for i, p in enumerate(pos):
        js = np.array(kd_tree.query_ball_point(p, connection_range, return_sorted=True, workers=-1))
        js = js[js > i]
        edges += [[i, j] for j in js]
    return np.array(edges)


def get_distances(node_positions: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Calculates the euclidean distances between connected nodes"""
    pos_i = node_positions[edges[:, 0]]
    pos_j = node_positions[edges[:, 1]]
    return np.linalg.norm(pos_i - pos_j, axis=1)


def get_fill_state_encodings(fill_states: np.ndarray) -> np.ndarray:
    """Encodes the fill_state as one-hot encoding."""
    filled = [1.0, 0.0]
    not_filled = [0.0, 1.0]
    return np.array([filled if fs else not_filled for fs in fill_states])


def get_data_file_path(raw_file_path: str, output_dir: str, time_step: int):
    """Returns a unique file path to store a data object"""
    _, raw_file_name = osp.split(raw_file_path)
    study_name, _ = osp.splitext(raw_file_name)
    data_file_name = f"data_{study_name}_{str(time_step).zfill(3)}.pt"
    data_file_path = osp.join(output_dir, data_file_name)
    return data_file_path


def process_raw_file(raw_file_path: str, output_dir: str, connection_range: float, time_step_size: float):
    """Process a raw study csv-file into multiple graph data objects stored at output_dir"""
    df_study: pd.DataFrame = pd.read_csv(raw_file_path)

    # todo: store positions so that they can be read directly from the dataframe
    node_positions = np.array([parse_to_float_list(p) for p in df_study.position])
    edge_list = get_edges(node_positions, connection_range)
    distances = get_distances(node_positions, edge_list)
    fill_states = get_fill_states(df_study.fill_time.to_numpy(), time_step_size)

    for t, _ in enumerate(fill_states[: -1]):
        node_attributes = get_fill_state_encodings(fill_states[t])
        target_node_attributes = get_fill_state_encodings(fill_states[t + 1])

        data = Data(
            x=torch.tensor(node_attributes, dtype=torch.float),
            y=torch.tensor(target_node_attributes, dtype=torch.float),
            edge_index=torch.from_numpy(edge_list).T,
            edge_weight=torch.tensor(distances, dtype=torch.float),
            pos=torch.from_numpy(node_positions)
        )
        data_file_path = get_data_file_path(raw_file_path, output_dir, t)
        torch.save(data, data_file_path)
