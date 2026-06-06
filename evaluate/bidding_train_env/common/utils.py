"""Shared preprocessing utilities used by the offline-evaluation pipeline."""

from __future__ import annotations

import os
import pickle
from typing import Iterable

import numpy as np
import pandas as pd


_EPS = 1e-10


def normalize_state(training_data: pd.DataFrame, state_dim: int, normalize_indices: Iterable[int]):
    """Min-max normalise selected dimensions of state / next_state in-place.

    Parameters
    ----------
    training_data:
        DataFrame with ``state`` and ``next_state`` columns containing tuples
        of length ``state_dim``.
    state_dim:
        Total dimensionality of each state vector.
    normalize_indices:
        Indices of the state dimensions that should be min-max normalised.

    Returns
    -------
    dict
        Per-dimension normalisation statistics for the requested indices.
    """
    normalize_indices = list(normalize_indices)
    state_columns = [f"state{i}" for i in range(state_dim)]
    next_state_columns = [f"next_state{i}" for i in range(state_dim)]

    for i, (s_col, ns_col) in enumerate(zip(state_columns, next_state_columns)):
        training_data[s_col] = training_data["state"].apply(
            lambda x, i=i: x[i] if x is not None and not np.isnan(x).any() else 0.0
        )
        training_data[ns_col] = training_data["next_state"].apply(
            lambda x, i=i: x[i] if x is not None and not np.isnan(x).any() else 0.0
        )

    stats = {
        i: {
            "min": training_data[state_columns[i]].min(),
            "max": training_data[state_columns[i]].max(),
            "mean": training_data[state_columns[i]].mean(),
            "std": training_data[state_columns[i]].std(),
        }
        for i in normalize_indices
    }

    for s_col, ns_col in zip(state_columns, next_state_columns):
        idx = int(s_col.replace("state", ""))
        if idx in normalize_indices:
            mn, mx = stats[idx]["min"], stats[idx]["max"]
            training_data[f"normalize_{s_col}"] = (training_data[s_col] - mn) / (mx - mn + _EPS)
            training_data[f"normalize_{ns_col}"] = (training_data[ns_col] - mn) / (mx - mn + _EPS)
        else:
            training_data[f"normalize_{s_col}"] = training_data[s_col]
            training_data[f"normalize_{ns_col}"] = training_data[ns_col]

    training_data["normalize_state"] = training_data.apply(
        lambda row: tuple(row[f"normalize_{c}"] for c in state_columns), axis=1
    )
    training_data["normalize_nextstate"] = training_data.apply(
        lambda row: tuple(row[f"normalize_{c}"] for c in next_state_columns), axis=1
    )
    return stats


def normalize_reward(training_data: pd.DataFrame, reward_type: str = "reward") -> pd.Series:
    """Min-max normalise the ``reward_type`` column of ``training_data``."""
    column = training_data[reward_type]
    span = column.max() - column.min() + _EPS
    training_data["normalize_reward"] = (column - column.min()) / span
    return training_data["normalize_reward"]


def save_normalize_dict(normalize_dict: dict, save_dir: str) -> None:
    """Pickle a normalisation dictionary to ``save_dir/normalize_dict.pkl``."""
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "normalize_dict.pkl"), "wb") as f:
        pickle.dump(normalize_dict, f)
