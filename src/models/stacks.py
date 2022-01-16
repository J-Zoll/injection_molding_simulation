from torch import nn
from typing import Iterable

from graph import Graph


class SequentialStack(nn.Module):
    def __init__(
        self,
        modules: Iterable[nn.Module],
        add_residual_connections: bool = True
    ):
        super().__init__()

        if add_residual_connections:
            modules = [ResidualConnection(m) for m in modules]

        self.model = nn.Sequential(modules)


    def forward(self, graph: Graph):
        return self.model(graph)



class ResidualConnection(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, graph: Graph):
        out = self.module(graph)
        return graph.sum_attributes(out)
