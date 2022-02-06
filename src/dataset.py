from torch_geometric.data import Dataset
import pandas as pd
import os
from tqdm import tqdm
from typing import Union, List, Tuple
from data_processing import preprocessing
from data_processing.util import parse_to_float_list
from torch import LongTensor, Tensor
import shutil
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
        self.clear_processed_dir()

        for raw_file_path in tqdm(self.raw_paths, desc="process studies"):
            preprocessing.process_raw_file(
                raw_file_path,
                self.processed_dir,
                self.connection_range,
                self.time_step_size
            )

    def len(self):
        file_names = os.listdir(self.processed_dir)
        data_file_names = [fn for fn in file_names if fn.startswith("data")]
        return len(data_file_names)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))

    def clear_processed_dir(self):
        """Removes all files that get generated during processing"""
        print("Clearing processed_dir..")

        # list files to delete
        files_to_delete = ["pre_filter.pt", "pre_transform.pt"]
        for fn in os.listdir(self.processed_dir):
            if fn.startswith("data"):
                files_to_delete.append(fn)

        # delete files
        for fn in files_to_delete:
            fp = os.path.join(self.processed_dir, fn)
            if os.path.isfile(fp):
                os.remove(fp)

        print("Finished clearing!")
