"""Base interface for offline-evaluation bidding strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseBiddingStrategy(ABC):
    """Base bidding strategy interface."""

    def __init__(self, budget: float = 100.0, name: str = "BaseStrategy", cpa: float = 2.0, category: int = 1):
        """
        Parameters
        ----------
        budget:
            Advertiser's budget for a delivery period.
        name:
            Human-readable identifier for the strategy implementation.
        cpa:
            CPA constraint of the advertiser.
        category:
            Index of the advertiser's industry category.
        """
        self.budget = budget
        self.remaining_budget = budget
        self.name = name
        self.cpa = cpa
        self.category = category

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state at the start of a new delivery period."""

    def bidding(
        self,
        timeStepIndex,
        pValues,
        pValueSigmas,
        historyPValueInfo,
        historyBid,
        historyAuctionResult,
        historyImpressionResult,
        historyLeastWinningCost,
    ):
        """Produce the bids for all impression opportunities in the current period.

        Parameters
        ----------
        timeStepIndex:
            Index of the current decision step.
        pValues, pValueSigmas:
            Predicted conversion probability and its uncertainty for each impression.
        historyPValueInfo, historyBid, historyAuctionResult,
        historyImpressionResult, historyLeastWinningCost:
            Per-step history for previous decision steps.
        """
        raise NotImplementedError

    def access_value(
        self,
        action,
        timeStepIndex,
        pValues,
        pValueSigmas,
        historyPValueInfo,
        historyBid,
        historyAuctionResult,
        historyImpressionResult,
        historyLeastWinningCost,
    ):
        raise NotImplementedError
