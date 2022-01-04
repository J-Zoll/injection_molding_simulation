"""A Multi Layer Perceptron module"""

from typing import Iterable

import torch
from torch import nn


class MLP (nn.Module):
    """A multi-layer perceptron module."""

    def __init__(
        self,
        layer_sizes: Iterable[int],
        activation: nn.Module = nn.ReLU(),
        activate_final: bool = True
    ):
        """Constructs an MLP.
        Args:
        layer_sizes: Sequence of layer sizes
                     e.g. [<input_size>, <hidden_size_1>, ... , <output_size>].
        activation: Activation function to apply between linear layers. Defaults to ReLU.
        activate_final: Whether or not to activate the final layer of the MLP. Defaults to True.
        """
        super().__init__()

        input_size, *output_sizes = layer_sizes
        self._input_size = input_size

        layers = []
        for output_size in output_sizes:
            layers.append(
                nn.Linear(
                    in_features=input_size,
                    out_features=output_size
                )
            )
            layers.append(activation)
            input_size = output_size

        if not activate_final:
            layers.pop()

        self._layers = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor):
        """Defines the computation performed at every call.
        Args:
        x: Sample tensor with shape [<batch_size>, <input_size>]
        """

        if any([
            len(x.shape) != 2,
            x.shape[-1] != self._input_size
        ]):
            raise ValueError(f"Sample tensor must have shape [<batch_size>, {self._input_size}]")

        return self._layers(x)
