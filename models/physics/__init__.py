"""
Physics-informed modules for molecular conformation generation.
Implements differentiable quantum mechanics (xTB) for Hamiltonian-Informed Schrödinger Bridge.
"""

from .dxtb_interface import DXTBForceField, compute_xtb_forces, compute_xtb_energy
from .gamma_scheduler import GammaScheduler, get_gamma_schedule

__all__ = [
    'DXTBForceField',
    'compute_xtb_forces',
    'compute_xtb_energy',
    'GammaScheduler',
    'get_gamma_schedule'
]
