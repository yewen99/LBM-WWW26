"""Offline-evaluation data loader.

Adapted from
https://github.com/alimama-tech/NeurIPS_Auto_Bidding_AIGB_Track_Baseline.
"""

from __future__ import annotations

import os
import pickle
import warnings

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")


class TestDataLoader:
    """Load AuctionNet evaluation traffic and group it by (period, advertiser)."""

    def __init__(self, file_path: str = "./data/log.csv"):
        self.file_path = file_path
        self.raw_data_path = os.path.join(os.path.dirname(file_path), "raw_data.pickle")
        self.raw_data = self._get_raw_data()
        self.keys, self.test_dict = self._get_test_data_dict()

    def _get_raw_data(self) -> pd.DataFrame:
        """Read raw data from a pickle cache, regenerating it from CSV if missing."""
        if os.path.exists(self.raw_data_path):
            with open(self.raw_data_path, "rb") as f:
                return pickle.load(f)

        df = pd.read_csv(self.file_path)
        with open(self.raw_data_path, "wb") as f:
            pickle.dump(df, f)
        return df

    def _get_test_data_dict(self):
        """Group raw data by ``(deliveryPeriodIndex, advertiserNumber)``."""
        grouped = self.raw_data.sort_values("timeStepIndex").groupby(
            ["deliveryPeriodIndex", "advertiserNumber"]
        )
        data_dict = {key: group for key, group in grouped}
        return list(data_dict.keys()), data_dict

    def mock_data(self, key):
        """Build the test inputs for a single advertiser/period."""
        data = self.test_dict[key]
        p_values = data.groupby("timeStepIndex")["pValue"].apply(list).apply(np.array).tolist()
        p_value_sigmas = data.groupby("timeStepIndex")["pValueSigma"].apply(list).apply(np.array).tolist()
        least_winning_costs = data.groupby("timeStepIndex")["leastWinningCost"].apply(list).apply(np.array).tolist()

        num_time_steps = len(p_values)
        budget = data["budget"].iloc[0]
        cpa = data["CPAConstraint"].iloc[0]
        category = data["advertiserCategoryIndex"].iloc[0]
        return num_time_steps, p_values, p_value_sigmas, least_winning_costs, budget, cpa, category
