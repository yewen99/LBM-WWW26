"""Offline evaluation environment for AuctionNet.

Adapted from
https://github.com/alimama-tech/NeurIPS_Auto_Bidding_AIGB_Track_Baseline.
"""

from __future__ import annotations

import numpy as np


class OfflineEnv:
    """Simulate an advertising auction environment."""

    def __init__(self, min_remaining_budget: float = 0.01):
        """
        Parameters
        ----------
        min_remaining_budget:
            Minimum remaining budget below which the advertiser is considered
            unable to bid further.
        """
        self.min_remaining_budget = min_remaining_budget

    def simulate_ad_bidding(
        self,
        pValues: np.ndarray,
        pValueSigmas: np.ndarray,
        bids: np.ndarray,
        leastWinningCosts: np.ndarray,
    ):
        """Simulate one round of auctions.

        Returns
        -------
        tick_value, tick_cost, tick_status, tick_conversion
            Per-impression values, charged costs, win indicator and
            (Bernoulli) conversion outcome.
        """
        tick_status = bids >= leastWinningCosts
        # The cost is charged for every won impression irrespective of whether
        # the user actually converted afterwards.
        tick_cost = leastWinningCosts * tick_status
        values = np.random.normal(loc=pValues, scale=pValueSigmas)
        values = values * tick_status
        tick_value = np.clip(values, 0, 1)
        tick_conversion = np.random.binomial(n=1, p=tick_value)
        return tick_value, tick_cost, tick_status, tick_conversion
