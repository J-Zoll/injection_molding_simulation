import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def get_fill_state(y_out):
    """Transforms raw output (logit) into fill state (bool)"""
    return F.softmax(y_out, dim=1)[:, 0].round()


class Rollout:
    def __init__(
            self,
            model: torch.nn.Module,
            data: Data,
            num_steps=35
    ):
        self.model = model
        self.data = data
        self.num_steps = num_steps
        self.fill_states = [get_fill_state(self.data.x)]

        self.simulate()

    def simulate(self):
        current_x = self.data.x
        for _ in tqdm(range(self.num_steps)):
            with torch.no_grad():
                y_out = self.model(self.data)
                fs = get_fill_state(y_out)
                self.fill_states.append(fs)
                current_x = y_out
