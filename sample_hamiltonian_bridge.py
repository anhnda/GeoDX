"""
Sampling script for Hamiltonian-Informed Schrödinger Bridge.

Generates molecular conformations using physics-aligned diffusion.
"""

import os
import argparse
import yaml
import pickle
from easydict import EasyDict
from tqdm.auto import tqdm
import torch
from torch_geometric.data import DataLoader, Batch

from models.epsnet.hamiltonian_bridge import HamiltonianBridgeNetwork
from utils.datasets import ConformationDataset
from utils.transforms import *
from utils.misc import *
import numpy as np


def sample_molecules(model, test_set, config, device, num_samples=10):
    """
    Generate conformations for test molecules.

    Args:
        model: trained Hamiltonian Bridge model
        test_set: test dataset
        config: configuration
        device: torch device
        num_samples: number of conformations to generate per molecule

    Returns:
        results: dict of generated conformations
    """
    model.eval()
    results = []

    for idx in tqdm(range(len(test_set)), desc='Generating conformations'):
        data = test_set[idx]

        # Prepare batch
        batch_list = [data] * num_samples
        batch = Batch.from_data_list(batch_list).to(device)

        # Sample conformations
        with torch.no_grad():
            pos_gen, pos_traj = model.sample(
                atom_type=batch.atom_type,
                bond_index=batch.edge_index,
                bond_type=batch.edge_type,
                batch=batch.batch,
                num_graphs=num_samples,
                n_steps=config.sampling.get('n_steps', model.num_timesteps),
                use_warm_start=config.sampling.get('use_warm_start', True),
                device=device
            )

        # Store results
        n_atoms = data.num_nodes
        pos_samples = pos_gen.view(num_samples, n_atoms, 3).cpu().numpy()

        results.append({
            'smiles': data.smiles if hasattr(data, 'smiles') else None,
            'atom_types': data.atom_type.cpu().numpy(),
            'positions_gt': data.pos.cpu().numpy(),
            'positions_gen': pos_samples,
            'num_atoms': n_atoms
        })

    return results


def evaluate_samples(results):
    """
    Evaluate generated conformations.

    Computes RMSD to ground truth and other metrics.
    """
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment

    rmsds = []

    for result in results:
        pos_gt = result['positions_gt']
        pos_gen = result['positions_gen']  # (num_samples, n_atoms, 3)

        # Compute RMSD for each sample
        sample_rmsds = []
        for pos_sample in pos_gen:
            # Align and compute RMSD
            # Simple RMSD (without alignment)
            rmsd = np.sqrt(np.mean((pos_sample - pos_gt) ** 2))
            sample_rmsds.append(rmsd)

        # Best RMSD (closest to ground truth)
        best_rmsd = min(sample_rmsds)
        rmsds.append(best_rmsd)

    metrics = {
        'mean_rmsd': np.mean(rmsds),
        'median_rmsd': np.median(rmsds),
        'std_rmsd': np.std(rmsds),
        'min_rmsd': np.min(rmsds),
        'max_rmsd': np.max(rmsds)
    }

    return metrics, rmsds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (default: load from checkpoint)')
    parser.add_argument('--test_set', type=str, required=True,
                       help='Path to test dataset')
    parser.add_argument('--output', type=str, default='./results_hamiltonian',
                       help='Output directory')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of conformations per molecule')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_mols', type=int, default=None,
                       help='Number of molecules to sample (default: all)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device)

    # Load config
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = EasyDict(yaml.safe_load(f))
    else:
        config = ckpt['config']

    print("="  * 80)
    print("Hamiltonian-Informed Schrödinger Bridge - Sampling")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Physics enabled: {config.model.get('use_physics', True)}")
    print(f"xTB method: {config.model.get('xtb_method', 'GFN2-xTB')}")
    print(f"Gamma schedule: {config.model.get('gamma_schedule', 'cosine')}")
    print(f"Warm-start: {config.sampling.get('use_warm_start', True)}")
    print(f"Num samples per molecule: {args.num_samples}")
    print("=" * 80)

    # Load test dataset
    print(f"Loading test dataset: {args.test_set}")
    transforms = CountNodesPerGraph()
    test_set = ConformationDataset(args.test_set, transform=transforms)

    if args.num_mols is not None:
        test_set = test_set[:args.num_mols]
        print(f"Using {args.num_mols} molecules")
    else:
        print(f"Using all {len(test_set)} molecules")

    # Build model
    print("Building model...")
    config.model.device = args.device
    model = HamiltonianBridgeNetwork(config.model).to(args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    print(f"Model loaded from iteration: {ckpt.get('iteration', 'unknown')}")

    # Generate samples
    print("\nGenerating conformations...")
    results = sample_molecules(
        model, test_set, config, args.device, num_samples=args.num_samples
    )

    # Save results
    results_path = os.path.join(args.output, 'generated_conformations.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to: {results_path}")

    # Evaluate
    print("\nEvaluating samples...")
    metrics, rmsds = evaluate_samples(results)

    print("\nMetrics:")
    print("-" * 40)
    for key, value in metrics.items():
        print(f"{key:20s}: {value:.4f} Å")
    print("-" * 40)

    # Save metrics
    metrics_path = os.path.join(args.output, 'metrics.yml')
    with open(metrics_path, 'w') as f:
        yaml.dump(metrics, f)
    print(f"Metrics saved to: {metrics_path}")

    # Save RMSD distribution
    rmsd_path = os.path.join(args.output, 'rmsds.npy')
    np.save(rmsd_path, np.array(rmsds))
    print(f"RMSD distribution saved to: {rmsd_path}")

    print("\nDone!")
