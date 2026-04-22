"""
Compatibility wrapper for the distribution-metrics scenes.

The original multi-scene file has been split into per-scene modules:

- mmd_distribution_difference_scene.py
- silhouette_weak_separation_scene.py
- softmax_probability_budget_scene.py
"""

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from mmd_distribution_difference_scene import (
    MMDDistributionDifferenceScene,
)
from silhouette_weak_separation_scene import (
    SilhouetteWeakSeparationScene,
)
from softmax_probability_budget_scene import (
    SoftmaxProbabilityBudgetScene,
)

__all__ = [
    "MMDDistributionDifferenceScene",
    "SilhouetteWeakSeparationScene",
    "SoftmaxProbabilityBudgetScene",
]
