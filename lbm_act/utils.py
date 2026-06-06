"""Utility functions for LBM-Act."""

from __future__ import annotations

import logging
import random
import sys
from typing import Sequence

import numpy as np
import torch
from torch import nn


DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config_logging(log_file: str = "main.log") -> None:
    """Configure global logging with both stdout and file handlers."""
    date_format = "%Y-%m-%d %H:%M:%S"
    log_format = "%(asctime)s: [%(levelname)s]: %(message)s"

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler], force=True)


class Squeeze(nn.Module):
    """Tiny utility module used by :func:`mlp` for output squeezing."""

    def __init__(self, dim: int | None = None):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


def mlp(
    dims: Sequence[int],
    activation: type[nn.Module] = nn.ReLU,
    output_activation: type[nn.Module] | None = None,
    squeeze_output: bool = False,
) -> nn.Sequential:
    """Build a simple multi-layer perceptron.

    Args:
        dims: Sequence of layer sizes ``(input, hidden..., output)``; at
            least 2 entries are required.
        activation: Hidden activation class.
        output_activation: Optional activation class applied to the output.
        squeeze_output: If True, append a :class:`Squeeze` layer along the
            last dimension. Requires ``dims[-1] == 1``.
    """
    if len(dims) < 2:
        raise ValueError("MLP requires at least two dimensions (input and output).")

    layers: list[nn.Module] = []
    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    if squeeze_output:
        if dims[-1] != 1:
            raise ValueError("squeeze_output=True requires dims[-1] == 1")
        layers.append(Squeeze(-1))

    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net


def torchify(x: np.ndarray) -> torch.Tensor:
    """Convert a NumPy array to a float tensor on :data:`DEFAULT_DEVICE`."""
    t = torch.from_numpy(x)
    if t.dtype is torch.float64:
        t = t.float()
    return t.to(device=DEFAULT_DEVICE)
