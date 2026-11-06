"""洋底水深的参数化方法。"""

import numpy as np


def gdh1_depth_m(age_myr):
    """GDH1 板块冷却模型的洋底水深（m，正值向下；Stein & Stein 1992, Nature 359:123）。"""
    age = np.maximum(np.asarray(age_myr, float), 0.0)
    return np.where(age < 20.0, 2600.0 + 365.0 * np.sqrt(age), 5651.0 - 2473.0 * np.exp(-0.0278 * age))
