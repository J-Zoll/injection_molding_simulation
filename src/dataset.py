import os
import sys
from typing import Union, List, Tuple
import functools

import torch
from torch_geometric.data import Dataset
from tqdm import tqdm

import preprocessing
from config import Config


class InjectionMoldingDataset(Dataset):
    def __init__(self, connection_range: float, time_step_size: float, skip_processing=False):
        super().__dict__["connection_range"] = connection_range
        super().__dict__["time_step_size"] = time_step_size
        super().__dict__["skip_processing"] = skip_processing
        super().__init__(Config.DATA_DIR)
        if not self.skip_processing:
            print('Processing...', file=sys.stderr)
            self.process()
            print('Done!', file=sys.stderr)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return [fn for fn in os.listdir(self.raw_dir) if fn.startswith("plate")]
    
    @property
    def processed_file_names(self):
        return sorted([fn for fn in os.listdir(self.processed_dir) if fn.startswith("data")])

    def download(self):
        pass

    def process(self):
        self.skip_processing = True
        self.clear_processed_dir()
        processing_function = functools.partial(
            preprocessing.process_raw_file,
            output_dir=self.processed_dir,
            connection_range=self.connection_range,
            time_step_size=self.time_step_size
        )
        for rp in tqdm(self.raw_paths):
            processing_function(rp)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(self.processed_paths[idx])

    def clear_processed_dir(self):
        """Removes all files that get generated during processing"""
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
