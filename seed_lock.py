"""
seed_lock.py
Drop this file into your project root (same folder as stage4_*.py,
stage4b_*.py, cca_analysis.py) and add these two lines to the TOP of
every stage script, before any model/data code runs:

    from seed_lock import set_seed
    set_seed(42)

This kills the two most common sources of "same script, different
numbers": unseeded weight init and unseeded data shuffling/sampling.
"""
import os
import random
import numpy as np


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Slower but removes cuDNN's own run-to-run randomness.
        # Comment these two lines out only if you need training speed
        # more than exact reproducibility.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    print(f"[seed_lock] All RNGs fixed to seed={seed}")