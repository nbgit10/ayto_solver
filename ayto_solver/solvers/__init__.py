"""Solvers for AYTO matching problem."""
from .graph_solver import GraphSolver

# MIP solvers require python-mip (AMD64 / Docker only). Import lazily.
try:
    from .mip_solver import MIPSolver
    from .mip_multi_solver import MIPMultiSolver
except ImportError:
    MIPSolver = None  # type: ignore[assignment,misc]
    MIPMultiSolver = None  # type: ignore[assignment,misc]

__all__ = ["MIPSolver", "MIPMultiSolver", "GraphSolver"]
