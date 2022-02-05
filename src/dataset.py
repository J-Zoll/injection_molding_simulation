from torch_geometric.data import Dataset, Data
import pandas as pd
import os
from tqdm import tqdm
from typing import Union, List, Tuple
from data_processing import preprocessing
from data_processing.util import parse_to_float_list
from torch import LongTensor, Tensor
import torch


class InjectionMoldingDataset(Dataset):
    def __init__(self, root: str, connection_range: float, time_step_size: float, skip_processing=False):
        super().__dict__["connection_range"] = connection_range
        super().__dict__["skip_processing"] = skip_processing
        super().__dict__["time_step_size"] = time_step_size
        super().__init__(root)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        NUM_STUDIES = 50
        file_names = [f"plate_{str(n).zfill(6)}" for n in range(NUM_STUDIES)]
        return file_names
    
    @property
    def processed_file_names(self):
        if self.skip_processing:
            return [fn for fn in os.listdir(self.processed_dir) if fn.startswith("data")]
        # trigger reprocessing
        return []

    def download(self):
        pass

    def process(self):
        data_counter = 0
        for path_to_raw_study_data in tqdm(self.raw_paths, desc="process studies"):
            df_study = pd.read_csv(f"{path_to_raw_study_data}.csv")

            node_positions = [parse_to_float_list(p) for p in df_study.position]
            
            # calculate edges and edge_distances
            edge_list = preprocessing.calculate_edges(node_positions, self.connection_range)
            edge_index = LongTensor(edge_list).T
            elementwise_distances = preprocessing.calculate_elementwise_distances(node_positions, edge_list)
            distances = preprocessing.calculate_distances(node_positions, edge_list)

            # calculate fill_states
            fill_states = preprocessing.calculate_fill_states(self.time_step_size, df_study.fill_time)

            for t, _ in enumerate(fill_states[:-1]):
                # fill_state at the beginning of the time step
                old_fs = Tensor(preprocessing.encode_fill_state(fill_states[t]))
                node_attributes = old_fs

                # (prediction goal) fill_state at the end of the time step
                new_fs = Tensor(preprocessing.encode_fill_state(fill_states[t + 1]))
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
                    study=path_to_raw_study_data,
                    time=t * self.time_step_size
                )
                torch.save(data, os.path.join(self.processed_dir, f"data_{data_counter}.pt"))
                data_counter += 1

    def len(self):
        file_names = os.listdir(self.processed_dir)
        data_file_names = [fn for fn in file_names if fn.startswith("data")]
        return len(data_file_names)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
