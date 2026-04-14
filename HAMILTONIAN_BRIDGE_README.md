# Hamiltonian-Informed Schrödinger Bridge for Molecular Conformation Generation

## Overview

This implementation extends the original **GeoDiff** model with physics-informed components based on the research proposal: **"Physics-Aligned Geometric Diffusion for Molecular Conformation Generation: From Schrödinger Priors to Cross-Domain Generalization"**.

### Key Innovation

The core idea is to replace the standard Brownian motion reference process in diffusion models with a **Langevin dynamics-based Schrödinger Bridge** that incorporates quantum mechanical forces from xTB (extended Tight-Binding).

**Standard Diffusion SDE:**
```
dC_t = -0.5 * β(t) * C_t * dt + sqrt(β(t)) * dW_t
```

**Our Hamiltonian-Informed SDE:**
```
dC_t = [-0.5 * β(t) * C_t + γ(t) * f_xTB(C_t)] * dt + sqrt(β(t)) * dW_t
```

where:
- `f_xTB(C_t) = -∇_C V_xTB(C)` are the xTB forces
- `γ(t)` is a time-dependent weight (strong at high noise, weak at low noise)

## Architecture

### Four-Stage Pipeline

```
┌──────────────────┐
│  Stage 1:        │
│  Warm Start      │──> xTB geometry optimization
│  (xTB optimize)  │    provides good initialization
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│  Stage 2:        │
│  Langevin SB     │──> Physics-informed reference SDE
│  (physics drift) │    constrains path to PES
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│  Stage 3:        │
│  Bridge Network  │──> Equivariant GFN learns
│  (xTB→DFT)       │    residual correction
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│  Stage 4:        │
│  Output          │──> DFT-quality conformer
│  (sampling)      │
└──────────────────┘
```

### Key Components

1. **dxtb Interface** (`models/physics/dxtb_interface.py`)
   - Differentiable xTB force field
   - Computes energies and forces via autograd
   - Warm-start optimization

2. **Gamma Scheduler** (`models/physics/gamma_scheduler.py`)
   - Controls physics drift weight γ(t)
   - Multiple schedules: constant, linear, cosine, exponential, sigmoid
   - Adaptive scheduling based on prediction quality

3. **Hamiltonian Bridge Network** (`models/epsnet/hamiltonian_bridge.py`)
   - Main model architecture
   - Combines learned components with physics forces
   - SE(3) equivariant throughout
   - Learns universal xTB→DFT correction

## Installation

### Requirements

```bash
# Base requirements (from GeoDiff)
torch>=1.9.0
torch-geometric>=2.0.0
torch-scatter
torch-sparse
rdkit
easydict
pyyaml
tqdm

# Physics components (NEW)
dxtb  # Differentiable xTB
```

### Install dxtb

```bash
# Option 1: pip (if available)
pip install dxtb

# Option 2: from source
git clone https://github.com/grimme-lab/dxtb.git
cd dxtb
pip install -e .
```

**Note:** If dxtb is not available, the code includes a mock implementation for testing. For production use, install the actual dxtb library.

## Usage

### Training

Train the Hamiltonian-Informed Schrödinger Bridge on QM9:

```bash
python train_hamiltonian_bridge.py \
    --config configs/hamiltonian_bridge_qm9.yml \
    --device cuda \
    --logdir ./logs_hamiltonian
```

### Configuration

Key configuration parameters in `configs/hamiltonian_bridge_qm9.yml`:

```yaml
model:
  network: hamiltonian_bridge

  # Physics components
  use_physics: true              # Enable physics-informed diffusion
  xtb_method: GFN2-xTB          # xTB method (GFN1-xTB or GFN2-xTB)

  # Gamma scheduler
  gamma_schedule: cosine         # Schedule type
  gamma_start: 1.0              # γ at t=T (high noise)
  gamma_end: 0.01               # γ at t=0 (low noise)

  # Warm-start
  warm_start_steps: 100         # xTB optimization steps
  physics_weight: 0.5           # Physics vs learned weight
```

### Sampling

Generate conformations using the trained model:

```bash
python sample_hamiltonian_bridge.py \
    --checkpoint logs_hamiltonian/checkpoints/500000.pt \
    --test_set data/GEOM/QM9/test.pkl \
    --output ./results \
    --num_samples 10 \
    --device cuda
```

## Implementation Details

### 1. Physics-Informed Loss

The training loss includes physics forces in the forward diffusion:

```python
# Standard noise
pos_perturbed = pos * sqrt(α_t) + noise * sqrt(1 - α_t)

# Add physics drift (NEW)
gamma_t = gamma_scheduler(t)
_, physics_forces = physics_field(atom_type, pos_perturbed, batch)
pos_perturbed = pos_perturbed + gamma_t * physics_forces * (1 - α_t)

# Network learns to denoise this physics-perturbed state
```

### 2. Gamma Scheduling

Different schedules for γ(t):

- **Cosine** (recommended): Smooth decay, balances physics and learning
  ```
  γ(t) = γ_end + (γ_start - γ_end) * 0.5 * (1 + cos(πt/T))
  ```

- **Exponential**: Faster decay, more learning at low noise
  ```
  γ(t) = γ_start * (γ_end/γ_start)^(t/T)
  ```

- **Inverse-sqrt**: Stronger physics at high noise
  ```
  γ(t) ∝ 1/sqrt(T - t + 1)
  ```

### 3. Warm-Start Initialization

Instead of pure noise, initialize from xTB-optimized geometry:

```python
# Random initialization
pos_init = torch.randn(N, 3)

# Optimize with xTB
pos_warm, energy = warm_start_opt.optimize(atom_type, pos_init)

# Add small noise
pos_start = pos_warm + torch.randn_like(pos_warm) * sigma_T * 0.5
```

### 4. SE(3) Equivariance

All components preserve SE(3) equivariance:

- xTB forces: `f_xTB(RC) = R * f_xTB(C)` (equivariant by construction)
- Gamma scheduler: operates on scalars (invariant)
- Bridge network: uses GFN layers from GeoDiff (equivariant)

## Expected Results

### Zero-Shot Transfer (QM9 → Drugs)

The model trained **only on QM9** should outperform vanilla models trained on GEOM-Drugs:

| Method | Training Data | RMSD (Drugs) | COV-R (%) |
|--------|--------------|--------------|-----------|
| GeoDiff | QM9 | ~0.86 Å | ~40% |
| DiSCO | QM9 | ~0.75 Å | ~65% |
| **Hamiltonian Bridge** | **QM9** | **~0.60 Å (target)** | **~75% (target)** |
| GeoDiff | Drugs | ~0.62 Å | ~89% |

### Why It Should Transfer

1. **Universal Physics**: xTB errors are similar across molecular families
2. **Residual Learning**: Model learns small, local corrections (not global patterns)
3. **Distribution-Free**: xTB→DFT gap is independent of training data distribution

## File Structure

```
GeoDiffX/
├── models/
│   ├── physics/
│   │   ├── __init__.py
│   │   ├── dxtb_interface.py      # Differentiable xTB
│   │   └── gamma_scheduler.py     # γ(t) scheduling
│   └── epsnet/
│       ├── diffusion.py           # Original GeoDiff (unchanged)
│       ├── hamiltonian_bridge.py  # NEW: Hamiltonian Bridge
│       └── __init__.py            # Updated to include new model
├── configs/
│   └── hamiltonian_bridge_qm9.yml # Configuration file
├── train_hamiltonian_bridge.py     # Training script
├── sample_hamiltonian_bridge.py    # Sampling script
└── HAMILTONIAN_BRIDGE_README.md    # This file
```

## Ablation Studies

To understand the contribution of each component:

### 1. Physics vs No Physics

```yaml
# Disable physics (vanilla GeoDiff)
model:
  use_physics: false
```

### 2. Different Gamma Schedules

```yaml
# Try different schedules
model:
  gamma_schedule: constant  # or linear, cosine, exponential
```

### 3. Warm-Start vs Pure Noise

```yaml
# Disable warm-start
sampling:
  use_warm_start: false
```

### 4. Different Physics Weights

```yaml
# Adjust physics contribution
model:
  physics_weight: 0.1  # Less physics
  physics_weight: 0.9  # More physics
```

## Comparison with Other Methods

### vs Score Mix (Strategy 1)

| Feature | Score Mix | Hamiltonian Bridge |
|---------|-----------|-------------------|
| Physics at inference | ✅ Every step | ✅ Amortized in model |
| Inference cost | 10-50× slower | 1× (same as vanilla) |
| Training complexity | Moderate | Higher |
| Transfer ability | Good | Better (residual learning) |

### vs Elign (Strategy 2)

| Feature | Elign | Hamiltonian Bridge |
|---------|-------|-------------------|
| Physics source | MLFF rewards | xTB gradients |
| When physics applied | Post-training (RL) | During training (SDE) |
| Training stability | RL (unstable) | Supervised (stable) |
| Physics integration | External | Intrinsic (in SDE) |

### vs DiSCO (Strategy 3)

| Feature | DiSCO | Hamiltonian Bridge |
|---------|-------|-------------------|
| Reference process | Brownian motion | Langevin (physics) |
| Starting point | RDKit | xTB-optimized |
| Bridge target | Data → DFT | xTB → DFT |
| Transfer mechanism | Statistical | Physical (universal) |

## Troubleshooting

### Issue: `dxtb not found`

**Solution:** Install dxtb or use mock implementation (for testing only)

### Issue: Out of memory during training

**Solution:** Reduce batch size or disable physics during training:
```yaml
train:
  batch_size: 32  # Reduce from 64
```

### Issue: NaN in losses

**Solution:** Check gradient clipping and learning rate:
```yaml
train:
  max_grad_norm: 4.0  # Reduce from 8.0
  optimizer:
    lr: 0.0005  # Reduce from 0.001
```

### Issue: Poor cross-domain transfer

**Solution:**
1. Increase gamma at high noise: `gamma_start: 2.0`
2. Use more physics: `physics_weight: 0.7`
3. Train longer: `max_iters: 2000000`

## Citation

If you use this code, please cite both the original GeoDiff paper and the proposal:

```bibtex
@inproceedings{xu2022geodiff,
  title={GeoDiff: A Geometric Diffusion Model for Molecular Conformation Generation},
  author={Xu, Minkai and Yu, Lantao and Song, Yang and Shi, Chence and Ermon, Stefano and Tang, Jian},
  booktitle={ICLR},
  year={2022}
}

@article{hamiltonian_bridge_2026,
  title={Physics-Aligned Geometric Diffusion for Molecular Conformation Generation},
  author={[Your Name]},
  journal={arXiv preprint},
  year={2026}
}
```

## Future Work

1. **Real dxtb Integration**: Replace mock with actual dxtb library
2. **MACE/UMA Forces**: Use foundation MLFFs instead of xTB
3. **Adaptive Gamma**: Learn γ(t) schedule during training
4. **Protein Extension**: Extend to protein structure generation
5. **Multi-Scale**: Hierarchical coarse-graining + Hamiltonian bridge

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.

---

**Implementation Status:** ✅ Complete (with mock dxtb for testing)
**Production Ready:** ⚠️ Requires real dxtb installation
**Tested On:** PyTorch 1.12+, CUDA 11.3+
