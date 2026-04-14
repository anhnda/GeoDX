"""
Test script for Hamiltonian-Informed Schrödinger Bridge.

Verifies that all components work correctly.
"""

import torch
import numpy as np
from easydict import EasyDict

# Test imports
print("Testing imports...")
try:
    from models.physics import DXTBForceField, GammaScheduler, WarmStartOptimizer
    from models.epsnet.hamiltonian_bridge import HamiltonianBridgeNetwork
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    exit(1)


def test_gamma_scheduler():
    """Test gamma scheduler."""
    print("\n1. Testing GammaScheduler...")

    scheduler = GammaScheduler(
        schedule_type='cosine',
        num_timesteps=1000,
        gamma_start=1.0,
        gamma_end=0.01
    )

    # Test forward pass
    t = torch.tensor([0, 500, 999])
    gammas = scheduler(t)

    print(f"   γ(0) = {gammas[0]:.4f} (should be ~1.0)")
    print(f"   γ(500) = {gammas[1]:.4f}")
    print(f"   γ(999) = {gammas[2]:.4f} (should be ~0.01)")

    assert gammas[0] > gammas[1] > gammas[2], "Gamma should decrease"
    print("   ✓ GammaScheduler passed")


def test_dxtb_interface():
    """Test dxtb interface."""
    print("\n2. Testing DXTBForceField...")

    device = 'cpu'
    calc = DXTBForceField(method='GFN2-xTB', device=device)

    # Create a simple molecule (water-like)
    atom_types = torch.tensor([8, 1, 1], dtype=torch.long)  # O-H-H
    positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.96, 0.0, 0.0],
        [-0.24, 0.93, 0.0]
    ], dtype=torch.float32)

    # Compute energy and forces
    energy, forces = calc(atom_types, positions)

    print(f"   Energy shape: {energy.shape}")
    print(f"   Forces shape: {forces.shape}")
    print(f"   Energy: {energy.item():.6f}")
    print(f"   Max force: {forces.abs().max().item():.6f}")

    assert energy.shape == (1,), "Energy should be scalar per molecule"
    assert forces.shape == (3, 3), "Forces should match positions shape"
    print("   ✓ DXTBForceField passed")


def test_hamiltonian_bridge_network():
    """Test Hamiltonian Bridge Network."""
    print("\n3. Testing HamiltonianBridgeNetwork...")

    # Create minimal config
    config = EasyDict({
        'network': 'hamiltonian_bridge',
        'hidden_dim': 64,
        'num_convs': 2,
        'num_convs_local': 2,
        'edge_order': 2,
        'cutoff': 5.0,
        'smooth_conv': True,
        'mlp_act': 'relu',
        'edge_encoder': 'gaussian',
        'beta_schedule': 'sigmoid',
        'beta_start': 1e-7,
        'beta_end': 2e-3,
        'num_diffusion_timesteps': 100,  # Reduced for testing
        'use_physics': True,
        'xtb_method': 'GFN2-xTB',
        'gamma_schedule': 'cosine',
        'gamma_start': 1.0,
        'gamma_end': 0.01,
        'warm_start_steps': 10,  # Reduced for testing
        'physics_weight': 0.5,
        'device': 'cpu'
    })

    device = 'cpu'
    model = HamiltonianBridgeNetwork(config).to(device)

    # Create a simple molecule
    atom_type = torch.tensor([6, 6, 8], dtype=torch.long)  # C-C-O
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [2.25, 1.0, 0.0]
    ], dtype=torch.float32)
    bond_index = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=torch.long).t()
    bond_type = torch.tensor([1, 1, 1, 1], dtype=torch.long)  # Single bonds
    batch = torch.zeros(3, dtype=torch.long)
    time_step = torch.tensor([50], dtype=torch.long)

    # Test forward pass
    print("   Testing forward pass...")
    edge_inv_global, edge_inv_local = model(
        atom_type=atom_type,
        pos=pos,
        bond_index=bond_index,
        bond_type=bond_type,
        batch=batch,
        time_step=time_step,
        return_edges=False
    )

    print(f"   Global edges: {edge_inv_global.shape}")
    print(f"   Local edges: {edge_inv_local.shape}")

    # Test loss computation
    print("   Testing loss computation...")
    loss, loss_global, loss_local = model.get_loss(
        atom_type=atom_type,
        pos=pos,
        bond_index=bond_index,
        bond_type=bond_type,
        batch=batch,
        num_nodes_per_graph=torch.tensor([3]),
        num_graphs=1,
        anneal_power=2.0,
        return_unreduced_loss=False
    )

    print(f"   Total loss: {loss.item():.6f}")
    print(f"   Global loss: {loss_global.item():.6f}")
    print(f"   Local loss: {loss_local.item():.6f}")

    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be Inf"
    print("   ✓ HamiltonianBridgeNetwork passed")


def test_sampling():
    """Test sampling (minimal)."""
    print("\n4. Testing sampling...")

    config = EasyDict({
        'network': 'hamiltonian_bridge',
        'hidden_dim': 32,  # Small for fast test
        'num_convs': 1,
        'num_convs_local': 1,
        'edge_order': 2,
        'cutoff': 5.0,
        'smooth_conv': True,
        'mlp_act': 'relu',
        'edge_encoder': 'gaussian',
        'beta_schedule': 'sigmoid',
        'beta_start': 1e-7,
        'beta_end': 2e-3,
        'num_diffusion_timesteps': 10,  # Very small for testing
        'use_physics': True,
        'xtb_method': 'GFN2-xTB',
        'gamma_schedule': 'cosine',
        'gamma_start': 1.0,
        'gamma_end': 0.01,
        'warm_start_steps': 5,
        'physics_weight': 0.5,
        'device': 'cpu'
    })

    device = 'cpu'
    model = HamiltonianBridgeNetwork(config).to(device)
    model.eval()

    # Simple molecule
    atom_type = torch.tensor([6, 6], dtype=torch.long)  # C-C
    bond_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()
    bond_type = torch.tensor([1, 1], dtype=torch.long)
    batch = torch.zeros(2, dtype=torch.long)

    print("   Generating sample (10 steps)...")
    with torch.no_grad():
        pos_gen, pos_traj = model.sample(
            atom_type=atom_type,
            bond_index=bond_index,
            bond_type=bond_type,
            batch=batch,
            num_graphs=1,
            n_steps=10,
            use_warm_start=False,  # Skip warm-start for speed
            device=device
        )

    print(f"   Generated positions: {pos_gen.shape}")
    print(f"   Trajectory length: {len(pos_traj)}")
    print(f"   Final positions:\n{pos_gen}")

    assert pos_gen.shape == (2, 3), "Should generate 2 atoms × 3 coords"
    assert len(pos_traj) == 10, "Should have 10 trajectory steps"
    print("   ✓ Sampling passed")


def test_equivariance():
    """Test SE(3) equivariance of physics forces."""
    print("\n5. Testing SE(3) equivariance...")

    device = 'cpu'
    calc = DXTBForceField(method='GFN2-xTB', device=device)

    # Original positions
    atom_types = torch.tensor([6, 6, 8], dtype=torch.long)
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [2.25, 1.0, 0.0]
    ], dtype=torch.float32)

    # Compute original forces
    _, forces_orig = calc(atom_types, pos)

    # Apply rotation
    angle = np.pi / 4
    R = torch.tensor([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ], dtype=torch.float32)
    pos_rot = torch.matmul(pos, R.t())

    # Compute rotated forces
    _, forces_rot = calc(atom_types, pos_rot)

    # Expected rotated forces
    forces_expected = torch.matmul(forces_orig, R.t())

    # Check equivariance
    diff = (forces_rot - forces_expected).abs().max()
    print(f"   Max difference: {diff.item():.8f}")

    # Allow some numerical error
    assert diff < 1e-4, f"Forces not equivariant: diff={diff.item()}"
    print("   ✓ SE(3) equivariance verified")


if __name__ == '__main__':
    print("=" * 60)
    print("Hamiltonian-Informed Schrödinger Bridge - Test Suite")
    print("=" * 60)

    try:
        test_gamma_scheduler()
        test_dxtb_interface()
        test_hamiltonian_bridge_network()
        test_sampling()
        test_equivariance()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print("\nImplementation is ready for use.")
        print("Note: Using mock xTB for testing. Install dxtb for production.")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
