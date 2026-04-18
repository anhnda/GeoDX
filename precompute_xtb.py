"""
Precompute xTB forces and geometries for Hamiltonian-Informed Schrödinger Bridge.

This script precomputes xTB energies and forces for the entire dataset offline,
so that training does not need to call xTB at every step.

Usage:
    python precompute_xtb.py configs/hamiltonian.yml --split train
    python precompute_xtb.py configs/hamiltonian.yml --split val
    python precompute_xtb.py configs/hamiltonian.yml --split test
    python precompute_xtb.py configs/hamiltonian.yml --split all --benchmark
    python precompute_xtb.py configs/hamiltonian.yml --compare --drugs_cache /path/to/drugs/cache

Output structure:
    {dataset_dir}/xtb_cache/
        train/
            00000000.pt   # per-sample cache
            00000001.pt
            ...
            meta.pt       # split-level statistics
        val/
            ...
        test/
            ...
        correction_comparison.pt   # QM9 vs Drugs analysis (if --compare)
"""

import os
import argparse
import yaml
import time
from easydict import EasyDict
from tqdm.auto import tqdm
from glob import glob

import torch
from torch_geometric.data import DataLoader

from utils.datasets import ConformationDataset
from utils.transforms import CountNodesPerGraph
from utils.misc import get_logger, seed_all
from models.physics import DXTBForceField, WarmStartOptimizer


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_cache_dir(config, split):
    """Return the cache directory for a given split, next to the dataset."""
    base = os.path.dirname(config.dataset.train)
    return os.path.join(base, 'xtb_cache', split)


def sample_path(cache_dir, idx):
    """Per-sample cache file path."""
    return os.path.join(cache_dir, f'{idx:08d}.pt')


def already_done(cache_dir, n_samples):
    """Count how many samples are already cached."""
    return sum(
        1 for i in range(n_samples)
        if os.path.exists(sample_path(cache_dir, i))
    )


# ---------------------------------------------------------------------------
# Per-sample precomputation
# ---------------------------------------------------------------------------

def precompute_sample(idx, data, physics_field, warm_start_opt,
                      cache_dir, config, logger, device):
    """
    Precompute and cache xTB data for a single molecule.

    Saves:
        pos_dft          : original DFT equilibrium geometry
        pos_warm         : xTB warm-start optimized geometry
        atom_type        : atom types
        xtb_energy_eq    : xTB energy at DFT geometry
        xtb_forces_eq    : xTB forces at DFT geometry
        xtb_energy_warm  : xTB energy at warm-start geometry
        xtb_forces_warm  : xTB forces at warm-start geometry  <- used in training
        correction       : C_DFT - C_xTB  (for Claim 2 analysis)
        correction_norm  : mean ||C_DFT - C_xTB|| per atom
    """
    out_path = sample_path(cache_dir, idx)
    if os.path.exists(out_path):
        return True   # already cached — skip

    try:
        atom_type = data.atom_type.to(device)
        pos_dft   = data.pos.to(device)
        batch     = torch.zeros(atom_type.size(0), dtype=torch.long, device=device)

        # ------------------------------------------------------------------
        # 1. xTB energy + forces at DFT geometry
        #    → used for correction analysis (C_DFT vs C_xTB)
        # ------------------------------------------------------------------
        with torch.no_grad():
            energy_eq, forces_eq = physics_field(atom_type, pos_dft, batch)

        # ------------------------------------------------------------------
        # 2. xTB warm-start geometry
        #    Optimize from DFT geometry + small noise → nearest xTB minimum
        # ------------------------------------------------------------------
        pos_init = pos_dft + torch.randn_like(pos_dft) * 0.1
        with torch.no_grad():
            pos_warm, _ = warm_start_opt.optimize(atom_type, pos_init, batch, device)

        # ------------------------------------------------------------------
        # 3. xTB energy + forces at warm-start geometry
        #    → used as physics guidance signal during training
        # ------------------------------------------------------------------
        with torch.no_grad():
            energy_warm, forces_warm = physics_field(atom_type, pos_warm, batch)

        # ------------------------------------------------------------------
        # 4. Correction vector  (DFT geometry − xTB geometry)
        #    → empirical evidence for Claim 2: "gap is consistent across domains"
        # ------------------------------------------------------------------
        correction      = pos_dft - pos_warm               # (N, 3)
        correction_norm = correction.norm(dim=-1).mean()   # scalar

        torch.save({
            # geometries
            'pos_dft':          pos_dft.cpu(),
            'pos_warm':         pos_warm.cpu(),
            'atom_type':        atom_type.cpu(),
            # xTB at DFT geometry
            'xtb_energy_eq':    energy_eq.cpu(),
            'xtb_forces_eq':    forces_eq.cpu(),
            # xTB at warm-start geometry  (primary training signal)
            'xtb_energy_warm':  energy_warm.cpu(),
            'xtb_forces_warm':  forces_warm.cpu(),
            # correction analysis
            'correction':       correction.cpu(),
            'correction_norm':  correction_norm.cpu(),
        }, out_path)

        return True

    except Exception as e:
        logger.warning(f'[Sample {idx:08d}] xTB failed: {e}')
        return False


# ---------------------------------------------------------------------------
# Full split precomputation
# ---------------------------------------------------------------------------

def precompute_split(config, split, device, logger, resume=True):
    """
    Precompute xTB data for an entire dataset split.

    Args:
        config  : EasyDict config (same YAML as train.py)
        split   : 'train' | 'val' | 'test'
        device  : torch device string
        logger  : logger instance
        resume  : if True, skip already-cached samples

    Returns:
        cache_dir : path to the cache directory
    """

    # Dataset
    dataset_path = getattr(config.dataset, split)
    logger.info(f'Loading {split} dataset from: {dataset_path}')
    transforms = CountNodesPerGraph()
    dataset    = ConformationDataset(dataset_path, transform=transforms)
    n_samples  = len(dataset)
    logger.info(f'{split} set size: {n_samples}')

    # Cache directory
    cache_dir = get_cache_dir(config, split)
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f'Cache directory: {cache_dir}')

    if resume:
        n_done = already_done(cache_dir, n_samples)
        logger.info(f'Already cached: {n_done}/{n_samples}')
        if n_done == n_samples:
            logger.info('All samples already cached. Skipping computation.')
            _save_meta(cache_dir, n_samples, split, logger)
            return cache_dir

    # Physics modules  (identical config keys as train.py)
    logger.info(f'Initializing xTB method: {config.model.get("xtb_method", "GFN2-xTB")}')
    physics_field  = DXTBForceField(
        method=config.model.get('xtb_method', 'GFN2-xTB'),
        device=device,
    )
    warm_start_opt = WarmStartOptimizer(
        method=config.model.get('xtb_method', 'GFN2-xTB'),
        max_steps=config.model.get('warm_start_steps', 100),
    )

    # Per-sample loop
    # Intentionally single-sample (not batched):
    # xTB call overhead is per-molecule, not per-batch; batching adds
    # bookkeeping complexity with no throughput benefit here.
    n_success = 0
    n_failed  = 0
    t_start   = time.time()

    for idx in tqdm(range(n_samples), desc=f'xTB precompute [{split}]'):
        data = dataset[idx]

        ok = precompute_sample(
            idx=idx,
            data=data,
            physics_field=physics_field,
            warm_start_opt=warm_start_opt,
            cache_dir=cache_dir,
            config=config,
            logger=logger,
            device=device,
        )

        if ok:
            n_success += 1
        else:
            n_failed += 1

        # Progress every 1000 samples
        if (idx + 1) % 1000 == 0:
            elapsed   = time.time() - t_start
            rate      = (idx + 1) / elapsed
            remaining = (n_samples - idx - 1) / max(rate, 1e-6)
            logger.info(
                f'[{split}] {idx+1}/{n_samples} | '
                f'Success {n_success} | Failed {n_failed} | '
                f'Rate {rate:.1f} mol/s | ETA {remaining/60:.1f} min'
            )

    logger.info(
        f'[{split}] Finished. '
        f'Success: {n_success}/{n_samples} | Failed: {n_failed}/{n_samples}'
    )

    # Aggregate metadata for analysis
    _save_meta(cache_dir, n_samples, split, logger)

    return cache_dir


# ---------------------------------------------------------------------------
# Metadata / statistics
# ---------------------------------------------------------------------------

def _save_meta(cache_dir, n_samples, split, logger):
    """
    Aggregate per-sample statistics into a single meta.pt file.

    The correction_norm statistics directly support Claim 2:
        "xTB→DFT correction is consistent across molecular families"
    Compare meta.pt from QM9 and Drugs caches to verify.
    """
    logger.info(f'Computing meta-statistics for {split}...')

    correction_norms = []
    force_norms_eq   = []
    force_norms_warm = []
    energies_eq      = []
    energies_warm    = []

    for idx in tqdm(range(n_samples), desc='Meta stats', leave=False):
        path = sample_path(cache_dir, idx)
        if not os.path.exists(path):
            continue
        d = torch.load(path, map_location='cpu')

        correction_norms.append(d['correction_norm'].item())
        force_norms_eq.append(d['xtb_forces_eq'].norm(dim=-1).mean().item())
        force_norms_warm.append(d['xtb_forces_warm'].norm(dim=-1).mean().item())

        if d['xtb_energy_eq'].numel() == 1:
            energies_eq.append(d['xtb_energy_eq'].item())
            energies_warm.append(d['xtb_energy_warm'].item())

    def _stats(lst):
        if not lst:
            return {}
        t = torch.tensor(lst)
        return {
            'mean': t.mean().item(),
            'std':  t.std().item(),
            'min':  t.min().item(),
            'max':  t.max().item(),
        }

    meta = {
        'split':           split,
        'n_samples':       n_samples,
        'n_cached':        len(correction_norms),
        'correction_norm': _stats(correction_norms),
        'force_norm_eq':   _stats(force_norms_eq),
        'force_norm_warm': _stats(force_norms_warm),
        'energy_eq':       _stats(energies_eq),
        'energy_warm':     _stats(energies_warm),
    }

    meta_path = os.path.join(cache_dir, 'meta.pt')
    torch.save(meta, meta_path)

    logger.info(
        f'[{split}] Correction norm — '
        f'mean={meta["correction_norm"].get("mean", float("nan")):.4f} Å  '
        f'std={meta["correction_norm"].get("std", float("nan")):.4f} Å'
    )
    logger.info(f'Meta saved: {meta_path}')


# ---------------------------------------------------------------------------
# Speed benchmark: online xTB vs cached disk load
# ---------------------------------------------------------------------------

def benchmark(config, device, logger, n_runs=20):
    """
    Measure online xTB time vs cached load time and print speedup.
    Run after precomputing the train split.
    """
    transforms = CountNodesPerGraph()
    dataset    = ConformationDataset(config.dataset.train, transform=transforms)
    cache_dir  = get_cache_dir(config, 'train')

    physics_field = DXTBForceField(
        method=config.model.get('xtb_method', 'GFN2-xTB'),
        device=device,
    )

    data      = dataset[0]
    atom_type = data.atom_type.to(device)
    pos       = data.pos.to(device)
    batch     = torch.zeros(atom_type.size(0), dtype=torch.long, device=device)

    logger.info(f'Benchmarking on molecule with {atom_type.size(0)} atoms, {n_runs} runs...')

    # Online xTB
    t0 = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            physics_field(atom_type, pos, batch)
    online_ms = (time.time() - t0) / n_runs * 1000

    # Cached load
    cached_path = sample_path(cache_dir, 0)
    if not os.path.exists(cached_path):
        logger.warning('No cached sample found at index 0. Run precompute first.')
        return

    t0 = time.time()
    for _ in range(n_runs):
        torch.load(cached_path, map_location='cpu')
    cached_ms = (time.time() - t0) / n_runs * 1000

    speedup = online_ms / max(cached_ms, 1e-6)

    logger.info('=' * 50)
    logger.info(f'Online xTB:   {online_ms:.1f} ms/mol')
    logger.info(f'Cached load:  {cached_ms:.1f} ms/mol')
    logger.info(f'Speedup:      {speedup:.0f}x')
    logger.info('=' * 50)


# ---------------------------------------------------------------------------
# Empirical Claim 2 validation: compare QM9 vs Drugs correction statistics
# ---------------------------------------------------------------------------

def compare_corrections(qm9_cache_dir, drugs_cache_dir, logger):
    """
    Compare xTB→DFT correction norms between QM9 and Drugs splits.

    Directly tests: "Is the xTB→DFT correction consistent across domains?"

    Interpretation:
        ratio ≈ 1.0  →  supports transfer assumption
        ratio >> 1.0 →  assumption is violated for Drugs
    """
    def load_corrections(cache_dir):
        files = sorted(glob(os.path.join(cache_dir, '????????.pt')))
        norms = []
        for f in tqdm(files, desc=f'Loading {os.path.basename(cache_dir)}'):
            d = torch.load(f, map_location='cpu')
            if 'correction_norm' in d:
                norms.append(d['correction_norm'].item())
        return torch.tensor(norms)

    logger.info('=' * 60)
    logger.info('Empirical Claim 2: QM9 vs Drugs correction comparison')
    logger.info('=' * 60)

    qm9_norms   = load_corrections(qm9_cache_dir)
    drugs_norms = load_corrections(drugs_cache_dir)

    logger.info(f'QM9   | n={len(qm9_norms):6d} | '
                f'mean={qm9_norms.mean():.4f} Å | '
                f'std={qm9_norms.std():.4f} Å')
    logger.info(f'Drugs | n={len(drugs_norms):6d} | '
                f'mean={drugs_norms.mean():.4f} Å | '
                f'std={drugs_norms.std():.4f} Å')

    ratio = qm9_norms.mean() / (drugs_norms.mean() + 1e-8)
    logger.info(f'Ratio QM9/Drugs: {ratio:.3f}  '
                f'(1.0 = identical correction scale = supports transfer)')

    out = {
        'qm9_correction_norms':   qm9_norms,
        'drugs_correction_norms': drugs_norms,
        'qm9_mean':    qm9_norms.mean().item(),
        'drugs_mean':  drugs_norms.mean().item(),
        'ratio':       ratio.item(),
    }
    out_path = 'correction_comparison.pt'
    torch.save(out, out_path)
    logger.info(f'Saved: {out_path}')
    logger.info('=' * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Precompute xTB forces for Hamiltonian Bridge training'
    )
    parser.add_argument(
        'config', type=str,
        help='Path to config YAML (same file used for train.py)'
    )
    parser.add_argument(
        '--split', type=str, default='all',
        choices=['train', 'val', 'test', 'all'],
        help='Which split to precompute (default: all)'
    )
    parser.add_argument(
        '--device', type=str, default='cpu',
        help='Device for xTB computation (cpu recommended, xTB is not GPU-native)'
    )
    parser.add_argument(
        '--logdir', type=str, default='./logs_precompute',
        help='Directory for log files'
    )
    parser.add_argument(
        '--no_resume', action='store_true',
        help='Recompute all samples even if cache already exists'
    )
    parser.add_argument(
        '--benchmark', action='store_true',
        help='After precomputing, benchmark online xTB vs cached load speed'
    )
    parser.add_argument(
        '--compare', action='store_true',
        help='Compare QM9 vs Drugs correction statistics (requires --drugs_cache)'
    )
    parser.add_argument(
        '--drugs_cache', type=str, default=None,
        help='Path to Drugs xtb_cache/train directory (for --compare)'
    )
    args = parser.parse_args()

    # Config  (identical loading pattern to train.py)
    with open(args.config, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    seed_all(config.train.seed)

    # Logging
    os.makedirs(args.logdir, exist_ok=True)
    logger = get_logger('precompute_xtb', args.logdir)

    logger.info('=' * 80)
    logger.info('xTB Precomputation — Hamiltonian-Informed Schrödinger Bridge')
    logger.info('=' * 80)
    logger.info(f'Config:      {args.config}')
    logger.info(f'Device:      {args.device}')
    logger.info(f'xTB method:  {config.model.get("xtb_method", "GFN2-xTB")}')
    logger.info(f'Warm steps:  {config.model.get("warm_start_steps", 100)}')
    logger.info(f'Split:       {args.split}')
    logger.info(f'Resume:      {not args.no_resume}')
    logger.info('=' * 80)

    # Precompute
    splits = ['train', 'val', 'test'] if args.split == 'all' else [args.split]

    for split in splits:
        logger.info(f'\n{"=" * 40}  {split}  {"=" * 40}')
        precompute_split(
            config=config,
            split=split,
            device=args.device,
            logger=logger,
            resume=not args.no_resume,
        )

    # Benchmark (optional)
    if args.benchmark:
        logger.info('\nRunning speed benchmark...')
        benchmark(config, args.device, logger)

    # Compare QM9 vs Drugs (optional — empirical Claim 2 validation)
    if args.compare:
        if args.drugs_cache is None:
            logger.warning('--drugs_cache not provided. Skipping comparison.')
        else:
            qm9_cache = get_cache_dir(config, 'train')
            compare_corrections(qm9_cache, args.drugs_cache, logger)

    logger.info('\nDone.')
    logger.info(
        '\nTo use cached forces in training, replace:\n'
        '    _, forces = self.physics_field(atom_type, pos, batch)\n'
        'with:\n'
        '    forces = batch.xtb_forces_warm   # free — loaded from disk\n'
    )