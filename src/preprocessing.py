import json
import os
import shutil
import sys
from tqdm import tqdm
from scipy.spatial import KDTree
import torch
from torch_geometric.data import Data
import os.path as osp
import numpy as np
from config import Config
import multiprocessing as mp
import functools


def get_fill_states(fill_times: np.ndarray, step_size: float) -> np.ndarray:
    """Calculates binary fill_states for a list of continuous node
       fill_times.
    """
    num_steps = np.max(fill_times).item() // step_size + 1
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
    """Process a raw study json-file into multiple graph data objects stored at output_dir"""
    with open(raw_file_path, "r") as json_file:
        study_dict = json.load(json_file)
    node_positions = np.array(study_dict["pos"])
    node_fill_times = np.array(study_dict["fill_time"])

    edge_list = get_edges(node_positions, connection_range)
    fill_states = get_fill_states(node_fill_times, time_step_size)
    distances = get_distances(node_positions, edge_list)

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


def process(input_dir: str, output_dir: str, connection_range: float, time_step_size: float):
    raw_file_paths = [osp.join(input_dir, fn) for fn in os.listdir(input_dir) if not fn.startswith(".")]
    for rfp in tqdm(raw_file_paths, desc="process"):
        process_raw_file(rfp, output_dir, connection_range, time_step_size)


def process_parallel(input_dir: str, output_dir: str, connection_range: float, time_step_size: float):
    raw_file_paths = [osp.join(input_dir, fn) for fn in os.listdir(input_dir) if not fn.startswith(".")]
    processing_function = functools.partial(
        process_raw_file,
        output_dir=output_dir,
        connection_range=connection_range,
        time_step_size=time_step_size
    )
    pool = mp.Pool(processes=8)
    for _ in tqdm(pool.imap_unordered(processing_function, raw_file_paths), total=len(raw_file_paths), desc="process"):
        pass
    pool.close()


def main():
    input_dir = Config.DATA_DIR / "raw"
    output_dir = Config.DATA_DIR / "processed"

    shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    connection_range = float(sys.argv[1])
    time_step_size = float(sys.argv[2])

    if mp.cpu_count() > 8:
        process(input_dir, output_dir, connection_range, time_step_size)
    else:
        process_parallel(input_dir, output_dir, connection_range, time_step_size)


if __name__ == "__main__":
    main()
