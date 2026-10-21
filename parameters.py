"""Parameters for the EAGLE router and the experiment, in one place.

The paper sets P, N and K (its Appendix A). Everything else here is this
repo's choice, either an addition to the paper or an experiment setting.
"""

import numpy as np

# EAGLE hyperparameters, from the paper's Appendix A

# Weight on the global rating in the combined rating; the local rating gets 1 - P
DEFAULT_P = 0.5

# Number of nearest training prompts used for a query's local rating (the paper's N)
DEFAULT_N_NEIGHBORS = 20

# Elo K-factor: how far a single match moves the two ratings
DEFAULT_K = 32

# Additions to the paper

# Largest gap between the two combined ratings that's predicted as a tie. The
# paper never predicts a tie, which a threshold of 0 matches.
DEFAULT_TIE_THRESHOLD = 0.0

# Elo starting rating for both models. The paper doesn't give one, so this is
# the conventional 1500. Only the gap between the two ratings affects
# predictions, so this value doesn't (unlike K).
DEFAULT_INIT_RATING = 1500.0

# Experiment settings

# Sample sizes, drawn from SPROUT's train/validation/test splits, and the seed
# used to draw them
DEFAULT_TRAIN_N = 1200
DEFAULT_VAL_N = 600
DEFAULT_TEST_N = 600
DEFAULT_SEED = 42

# Grid searched on the validation set: 21 evenly spaced values of P, and 0 plus
# 61 thresholds spaced evenly on a log scale from 1 to 1000
P_VALUES = np.linspace(0, 1, 21)  # (21,): 0.0, 0.05, ..., 1.0
TIE_THRESHOLD_VALUES = [0.0, *np.geomspace(1, 1000, 61)]  # 62 values
