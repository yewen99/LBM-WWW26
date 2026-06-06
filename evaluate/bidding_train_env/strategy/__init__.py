"""LBM bidding strategy package."""

from .lbm_bidding_strategy import LbmBiddingStrategy

# Backwards-compatible alias for older callers / scripts.
LBM_BiddingStrategy = LbmBiddingStrategy

__all__ = ["LbmBiddingStrategy", "LBM_BiddingStrategy"]
