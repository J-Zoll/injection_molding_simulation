
from torch import nn
from graph import Graph

class EncProDec(nn.Module):
    """Module with an Encoder-Processor-Decoder Architecture"""

    def __init__(
        self,
        encoder: nn.Module,
        processor: nn.Module,
        decoder: nn.Module
        ):
        super().__init__()
        self.model = nn.Sequential([
            encoder,
            processor,
            decoder
        ])


    def forward(self, graph: Graph):
        return self.model(graph)
        