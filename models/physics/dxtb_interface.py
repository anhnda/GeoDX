"""
Differentiable xTB (dxtb) interface for computing physics forces and energies.
This module enables backpropagation through quantum mechanics calculations.

Reference: Friede et al., "Fully Differentiable Extended Tight-Binding Hamiltonian", 2024
"""

import torch
import torch.nn as nn
import numpy as np

try:
    import dxtb
    DXTB_AVAILABLE = True
except ImportError:
    DXTB_AVAILABLE = False
    print("Warning: dxtb not available. Install with: pip install dxtb")


class DXTBForceField(nn.Module):
    """
    Wrapper for differentiable xTB force field calculations.
    Computes energies and forces (negative gradients) using dxtb.
    """

    def __init__(self, method='GFN2-xTB', device='cpu'):
        """
        Args:
            method: xTB method to use ('GFN1-xTB', 'GFN2-xTB')
            device: torch device
        """
        super().__init__()
        self.method = method
        self.device = device

        if not DXTB_AVAILABLE:
            raise ImportError("dxtb is required but not installed")

    def forward(self, atom_types, positions, batch=None):
        """
        Compute xTB energy and forces.

        Args:
            atom_types: (N,) atomic numbers
            positions: (N, 3) atomic coordinates in Angstrom
            batch: (N,) batch indices for multiple molecules

        Returns:
            energy: (n_graphs,) total energy in Hartree
            forces: (N, 3) forces in Hartree/Angstrom (negative of energy gradient)
        """
        if batch is None:
            batch = torch.zeros(atom_types.size(0), dtype=torch.long, device=self.device)

        n_graphs = batch.max().item() + 1
        energies = []
        all_forces = []

        for i in range(n_graphs):
            mask = (batch == i)
            mol_atom_types = atom_types[mask]
            mol_positions = positions[mask]

            # Compute energy and forces for this molecule
            energy, forces = self._compute_single_molecule(mol_atom_types, mol_positions)
            energies.append(energy)

            # Store forces in full tensor
            mol_forces = torch.zeros_like(positions)
            mol_forces[mask] = forces
            all_forces.append(mol_forces)

        total_energy = torch.stack(energies)
        total_forces = torch.stack(all_forces).sum(0)

        return total_energy, total_forces

    def _compute_single_molecule(self, atom_types, positions):
        """
        Compute energy and forces for a single molecule using dxtb.

        Args:
            atom_types: (n,) atomic numbers
            positions: (n, 3) coordinates

        Returns:
            energy: scalar energy
            forces: (n, 3) forces
        """
        # Enable gradient computation
        positions = positions.requires_grad_(True)

        # Convert atom types to atomic numbers if needed
        if atom_types.dtype == torch.long:
            atomic_numbers = atom_types.to(self.device)
        else:
            atomic_numbers = atom_types.long().to(self.device)

        # Compute energy using dxtb
        # Note: Actual dxtb API may vary, this is a placeholder
        # Real implementation would use: dxtb.Calculator(method=self.method)
        energy = self._compute_xtb_energy_mock(atomic_numbers, positions)

        # Compute forces as negative gradients
        if positions.grad is not None:
            positions.grad.zero_()

        energy.backward(retain_graph=True)
        forces = -positions.grad.clone()

        return energy, forces

    def _compute_xtb_energy_mock(self, atomic_numbers, positions):
        """
        Mock xTB energy computation.
        In production, this would call actual dxtb library.

        For now, use a simple molecular mechanics approximation.
        """
        # Simple harmonic potential as placeholder
        # E = sum of pairwise LJ-like interactions
        n_atoms = positions.size(0)
        energy = torch.tensor(0.0, device=positions.device, requires_grad=True)

        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                r = torch.norm(positions[i] - positions[j])
                # Simple LJ-like potential: E = 4ε[(σ/r)^12 - (σ/r)^6]
                sigma = 1.5  # Angstrom
                epsilon = 0.1  # Energy unit
                energy = energy + 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)

        return energy


def compute_xtb_forces(atom_types, positions, batch=None, method='GFN2-xTB', device='cpu'):
    """
    Convenience function to compute xTB forces.

    Args:
        atom_types: (N,) atomic numbers
        positions: (N, 3) coordinates
        batch: (N,) batch indices
        method: xTB method
        device: torch device

    Returns:
        forces: (N, 3) forces
    """
    calculator = DXTBForceField(method=method, device=device)
    _, forces = calculator(atom_types, positions, batch)
    return forces


def compute_xtb_energy(atom_types, positions, batch=None, method='GFN2-xTB', device='cpu'):
    """
    Convenience function to compute xTB energy.

    Args:
        atom_types: (N,) atomic numbers
        positions: (N, 3) coordinates
        batch: (N,) batch indices
        method: xTB method
        device: torch device

    Returns:
        energy: (n_graphs,) energies
    """
    calculator = DXTBForceField(method=method, device=device)
    energy, _ = calculator(atom_types, positions, batch)
    return energy


class WarmStartOptimizer:
    """
    Performs xTB geometry optimization to provide warm-start conformations.
    """

    def __init__(self, method='GFN2-xTB', max_steps=100, force_tol=1e-3):
        """
        Args:
            method: xTB method
            max_steps: maximum optimization steps
            force_tol: force convergence tolerance
        """
        self.method = method
        self.max_steps = max_steps
        self.force_tol = force_tol

    def optimize(self, atom_types, positions_init, batch=None, device='cpu'):
        """
        Optimize molecular geometry using xTB forces.

        Args:
            atom_types: (N,) atomic numbers
            positions_init: (N, 3) initial coordinates
            batch: (N,) batch indices
            device: torch device

        Returns:
            positions_opt: (N, 3) optimized coordinates
            energy: final energy
        """
        calculator = DXTBForceField(method=self.method, device=device)
        positions = positions_init.clone().detach().requires_grad_(False)

        # Simple gradient descent optimization
        lr = 0.01
        for step in range(self.max_steps):
            positions.requires_grad_(True)
            energy, forces = calculator(atom_types, positions, batch)

            # Check convergence
            max_force = torch.abs(forces).max()
            if max_force < self.force_tol:
                print(f"Converged at step {step}, max force: {max_force:.6f}")
                break

            # Update positions
            with torch.no_grad():
                # Move in direction of forces
                positions = positions + lr * forces
                positions = positions.detach()

        return positions, energy
