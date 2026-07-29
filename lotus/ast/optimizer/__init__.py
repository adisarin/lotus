"""Optimizer module for LOTUS LazyFrames."""

from .accuracy_optimizer import AccuracyInferenceOptimizer
from .base import BaseOptimizer
from .cascade import CascadeOptimizer
from .gepa_optimizer import GEPAOptimizer
from .predicate_pushdown import PredicatePushdownOptimizer
from .utils import PathEntry, PathToLF

DEFAULT_OPTIMIZERS: list[BaseOptimizer] = [PredicatePushdownOptimizer()]

__all__ = [
    "AccuracyInferenceOptimizer",
    "BaseOptimizer",
    "CascadeOptimizer",
    "DEFAULT_OPTIMIZERS",
    "GEPAOptimizer",
    "PathEntry",
    "PathToLF",
    "PredicatePushdownOptimizer",
]
