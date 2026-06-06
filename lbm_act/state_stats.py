"""Pre-computed state normalisation statistics for AuctionNet.

The means/stds below are computed once over the preprocessed AuctionNet
trajectories (16-dimensional state vector, see ``lbm_act/seq_dataset.py``)
and are reused at inference time so that the policy receives inputs with
the same distribution it was trained on.
"""

from __future__ import annotations

import numpy as np


# ---------------------------- Dense (AuctionNet) ---------------------------- #
DENSE_STATE_MEAN = np.array(
    [
        5.48876391e-01, 6.91904804e-01, 4.80044229e-02, 4.47875045e-02,
        1.17763952e-01, 4.87555661e-03, 4.76420127e-04, 5.72794009e-02,
        9.93989091e-02, 4.84664169e-03, 5.83001837e-04, 7.04144008e-02,
        4.99805521e-03, 1.01522635e+04, 2.86396864e+04, 1.91412327e+05,
    ]
)
DENSE_STATE_STD = np.array(
    [
        2.84053382e-01, 3.53000441e-01, 3.01172049e-02, 3.21944272e-02,
        3.07672391e-02, 1.92189715e-03, 8.29556557e-04, 9.36906833e-02,
        3.75196803e-02, 2.45325444e-03, 1.18077056e-03, 1.27290708e-01,
        2.48126164e-03, 5.73180055e+03, 1.67849786e+04, 1.52535424e+05,
    ]
)
DENSE_RTG_SCALE = 1500.0


# --------------------------- Sparse (AuctionNet-S) -------------------------- #
SPARSE_STATE_MEAN = np.array(
    [
        5.41854588e-01, 7.19698607e-01, 4.17500439e-02, 4.35970703e-02,
        9.91188952e-02, 4.82405201e-04, 4.61863046e-05, 5.29802530e-02,
        9.24203256e-02, 4.84138679e-04, 5.76074165e-05, 6.75957800e-02,
        4.98045765e-04, 1.02017857e+04, 2.88687230e+04, 1.95333666e+05,
    ]
)
SPARSE_STATE_STD = np.array(
    [
        2.84601949e-01, 3.27488061e-01, 2.76529743e-02, 3.31906076e-02,
        2.38985949e-02, 1.89047081e-04, 8.73831598e-05, 9.07318426e-02,
        2.65035680e-02, 2.45689550e-04, 1.26855462e-04, 1.23013225e-01,
        2.48356154e-04, 5.72176857e+03, 1.67729807e+04, 1.52914080e+05,
    ]
)
SPARSE_RTG_SCALE = 100.0


STATE_DIM = 16


def get_state_stats(sparse: bool) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(mean, std, rtg_scale)`` for the requested data variant."""
    if sparse:
        return SPARSE_STATE_MEAN, SPARSE_STATE_STD, SPARSE_RTG_SCALE
    return DENSE_STATE_MEAN, DENSE_STATE_STD, DENSE_RTG_SCALE
