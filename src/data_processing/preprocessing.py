from platform import node
from typing import Iterable, Tuple
import cleaning
from util import binary_aggregate, componentwise_distance, discretize
from graph import Graph
from torch import Tensor
import os
import pickle
from scipy.spatial import KDTree
from tqdm import tqdm


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
    for i, pos in tqdm(enumerate(node_positions), desc="calculate edges"):
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
    for i, j in tqdm(edges, desc="calculate distances"):
        p1 = node_positions[i]
        p2 = node_positions[j]
        distances.append(componentwise_distance(p1, p2))
    return distances


def encode_fill_state(fill_states: Iterable[bool]) -> Iterable[Tuple[float, float]]:
    """Encodes the fill_state trough one-hot encoding."""
    FILLED = [1.0, 0.0]
    NOT_FILLED = [0.0, 1.0]

    return [FILLED if fs else NOT_FILLED for fs in fill_states]


def convert_to_graph(path_to_study_dir: str, connection_range: float, time_step_size: float):
    """Converts study to a list of graphs and stores them with pickle in the studys directory"""

    study = cleaning.load_study(path_to_study_dir)

    node_positions = study.position
    edges = calculate_edges(node_positions.tolist(), connection_range)
    distances = calculate_distances(node_positions, edges)

    fill_times = study.fill_time
    fill_states = calculate_fill_states(time_step_size, fill_times)

    # make sure output dir exists
    output_dir = os.path.join(path_to_study_dir, "graphs")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # convert to Graph
    for step, fs in tqdm(enumerate(fill_states), desc="convert to graph"):
        enc_fs = encode_fill_state(fs)

        edge_attributes = Tensor(distances)
        node_attributes = Tensor(enc_fs)

        graph = Graph(
            edges,
            edge_attributes=edge_attributes,
            node_attributes=node_attributes,
            node_positions=node_positions
        )

        # store graph
        file_path = os.path.join(output_dir, f"graph_t{step}.pickle")
        with open(file_path, "wb") as pickle_file:
            pickle.dump(graph, pickle_file)


def convert_whole_dataset_to_graph(path_to_dataset: str, connection_range: float, time_step_size: float) -> None:
    file_names = os.listdir(path_to_dataset)
    file_paths = [os.path.join(path_to_dataset, fn) for fn in file_names]
    study_dir_paths = [dir for dir in file_paths if os.path.isdir(dir)]

    for sdp in study_dir_paths:
        convert_to_graph(sdp, connection_range, time_step_size)


def load_graphs(path_to_study_dir: str) -> Iterable[Graph]:
    graph_dir = os.path.join(path_to_study_dir, "graphs")
    pickle_names = os.listdir(graph_dir)
    pickle_paths = [os.path.join(graph_dir, pn) for pn in pickle_names]

    graphs = []
    for pp in pickle_paths:
        with open(pp, "rb") as file:
            graphs.append(pickle.load(file))
    
    return graphs
