"""
Training script for Hamiltonian-Informed Schrödinger Bridge.

This implements the physics-aligned geometric diffusion model
with xTB-informed Langevin dynamics.

Speed improvement over original:
    - xTB forces are loaded from precomputed cache (precompute_xtb.py)
      instead of calling DXTBForceField at every training step.
    - Expected speedup: ~100x per batch on drug-sized molecules.

Usage
-----
# Normal training (uses cache if available, falls back to online xTB):
python train_hamiltonian.py configs/hamiltonian.yml

# Resume:
python train_hamiltonian.py logs_hamiltonian/my_run --resume_iter 50000

Precompute cache first (recommended):
    python precompute_xtb.py configs/hamiltonian.yml --split all
"""

import os
import shutil
import argparse
import yaml
from easydict import EasyDict
from tqdm.auto import tqdm
from glob import glob

import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader, Data

from models.epsnet.hamiltonian_bridge import HamiltonianBridgeNetwork
from utils.datasets import ConformationDataset
from utils.transforms import CountNodesPerGraph
from utils.misc import get_logger, get_new_log_dir, seed_all, get_checkpoint_path
from utils.common import get_optimizer, get_scheduler


# ============================================================
#  Cached dataset wrapper
# ============================================================

class CachedXTBDataset(torch.utils.data.Dataset):
    """
    Wraps a ConformationDataset and attaches precomputed xTB data.

    Each sample gains two extra attributes:
        data.xtb_forces_warm  (N, 3)  xTB forces at warm-start geometry
        data.xtb_energy_warm  ()      xTB energy at warm-start geometry

    If a sample's cache file is missing, both are set to None and the
    training step falls back to online xTB (same behaviour as original).
    """

    def __init__(self, conformation_dataset, cache_dir):
        self.base      = conformation_dataset
        self.cache_dir = cache_dir
        self._warned   = False   # warn only once about missing cache

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data = self.base[idx]

        cache_path = os.path.join(self.cache_dir, f'{idx:08d}.pt')
        if os.path.exists(cache_path):
            cached = torch.load(cache_path, map_location='cpu')
            data.xtb_forces_warm = cached['xtb_forces_warm']   # (N, 3)
            data.xtb_energy_warm = cached['xtb_energy_warm']   # scalar
        else:
            if not self._warned:
                print(
                    f'[CachedXTBDataset] WARNING: cache missing for idx={idx}. '
                    f'Falling back to online xTB. '
                    f'Run precompute_xtb.py to generate cache.'
                )
                self._warned = True
            data.xtb_forces_warm = None
            data.xtb_energy_warm = None

        return data


def get_cache_dir(config, split):
    """Same logic as precompute_xtb.py — must stay in sync."""
    base = os.path.dirname(config.dataset.train)
    return os.path.join(base, 'xtb_cache', split)


def has_cache(config, split):
    """Return True if at least one cached file exists for this split."""
    cache_dir = get_cache_dir(config, split)
    return os.path.isdir(cache_dir) and bool(
        glob(os.path.join(cache_dir, '????????.pt'))
    )


def build_dataset(config, split, logger):
    """
    Build dataset for a given split.

    If precomputed xTB cache exists → wrap with CachedXTBDataset.
    Otherwise → plain ConformationDataset (online xTB, slow).
    """
    dataset_path = getattr(config.dataset, split)
    transforms   = CountNodesPerGraph()
    base_dataset = ConformationDataset(dataset_path, transform=transforms)

    cache_dir = get_cache_dir(config, split)
    if has_cache(config, split):
        logger.info(f'[{split}] Using precomputed xTB cache: {cache_dir}')
        return CachedXTBDataset(base_dataset, cache_dir), True
    else:
        logger.info(
            f'[{split}] No xTB cache found at {cache_dir}. '
            f'Falling back to online xTB (slow). '
            f'Run: python precompute_xtb.py {args_config} --split {split}'
        )
        return base_dataset, False


# ============================================================
#  Collate function — handles optional xtb_forces_warm field
# ============================================================

def collate_with_xtb(data_list):
    """
    Custom collate that stacks xtb_forces_warm across graphs.

    torch_geometric's default Batch.from_data_list handles standard
    fields. We manually stack xtb_forces_warm since it is a variable-
    size tensor (different N per molecule).
    """
    from torch_geometric.data import Batch

    # Separate out xTB fields before batching
    forces_list = []
    energy_list = []
    has_forces  = all(
        d.xtb_forces_warm is not None
        for d in data_list
        if hasattr(d, 'xtb_forces_warm')
    )

    if has_forces:
        for d in data_list:
            forces_list.append(d.xtb_forces_warm)
            energy_list.append(d.xtb_energy_warm)
            # Remove from data so Batch.from_data_list doesn't try to cat them
            d.xtb_forces_warm = None
            d.xtb_energy_warm = None

    batch = Batch.from_data_list(data_list)

    if has_forces:
        # Stack along atom dimension (matches batch.pos layout)
        batch.xtb_forces_warm = torch.cat(forces_list, dim=0)   # (N_total, 3)
        batch.xtb_energy_warm = torch.stack(energy_list, dim=0) # (B,)
    else:
        batch.xtb_forces_warm = None
        batch.xtb_energy_warm = None

    return batch


# ============================================================
#  Training step
# ============================================================

def train_step(model, batch, optimizer_global, optimizer_local, config, device):
    """Single training step."""
    model.train()
    optimizer_global.zero_grad()
    optimizer_local.zero_grad()

    batch = batch.to(device)

    # Pass cached xTB forces to get_loss so it doesn't recompute them.
    # If batch.xtb_forces_warm is None, get_loss falls back to online xTB.
    loss, loss_global, loss_local = model.get_loss(
        atom_type=batch.atom_type,
        pos=batch.pos,
        bond_index=batch.edge_index,
        bond_type=batch.edge_type,
        batch=batch.batch,
        num_nodes_per_graph=batch.num_nodes_per_graph,
        num_graphs=batch.num_graphs,
        anneal_power=config.train.anneal_power,
        return_unreduced_loss=False,
        # ---- NEW: pass precomputed forces ----
        precomputed_forces=getattr(batch, 'xtb_forces_warm', None),
    )

    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
    optimizer_global.step()
    optimizer_local.step()

    return {
        'loss':       loss.item(),
        'loss_global': loss_global.item(),
        'loss_local':  loss_local.item(),
        'grad_norm':   orig_grad_norm,
    }


# ============================================================
#  Validation loop
# ============================================================

def validate(model, val_loader, config, device):
    """Validation loop."""
    model.eval()
    sum_loss        = 0.0
    sum_loss_global = 0.0
    sum_loss_local  = 0.0
    count           = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            batch = batch.to(device)

            loss, loss_global, loss_local = model.get_loss(
                atom_type=batch.atom_type,
                pos=batch.pos,
                bond_index=batch.edge_index,
                bond_type=batch.edge_type,
                batch=batch.batch,
                num_nodes_per_graph=batch.num_nodes_per_graph,
                num_graphs=batch.num_graphs,
                anneal_power=config.train.anneal_power,
                return_unreduced_loss=False,
                precomputed_forces=getattr(batch, 'xtb_forces_warm', None),
            )

            sum_loss        += loss.item()        * batch.num_graphs
            sum_loss_global += loss_global.item() * batch.num_graphs
            sum_loss_local  += loss_local.item()  * batch.num_graphs
            count           += batch.num_graphs

    return {
        'loss':       sum_loss        / count,
        'loss_global': sum_loss_global / count,
        'loss_local':  sum_loss_local  / count,
    }


# ============================================================
#  Entry point
# ============================================================

# We need args.config before build_dataset prints its fallback message,
# so store it globally once parsed.
args_config = ''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config',         type=str,
                        help='Path to config file or resume directory')
    parser.add_argument('--device',       type=str, default='cuda')
    parser.add_argument('--resume_iter',  type=int, default=None)
    parser.add_argument('--logdir',       type=str, default='./logs_hamiltonian')
    parser.add_argument('--no_cache',     action='store_true',
                        help='Disable xTB cache and use online xTB (for debugging)')
    args = parser.parse_args()
    args_config = args.config   # used in build_dataset fallback message

    # ---- config ----
    resume = os.path.isdir(args.config)
    if resume:
        config_path  = glob(os.path.join(args.config, '*.yml'))[0]
        resume_from  = args.config
    else:
        config_path  = args.config

    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]
    seed_all(config.train.seed)

    # ---- logging ----
    if resume:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag='resume')
        os.symlink(
            os.path.realpath(resume_from),
            os.path.join(log_dir, os.path.basename(resume_from.rstrip('/')))
        )
    else:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name)
        shutil.copytree('./models', os.path.join(log_dir, 'models'))

    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = get_logger('train_hamiltonian', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)

    logger.info('=' * 80)
    logger.info('Hamiltonian-Informed Schrödinger Bridge Training')
    logger.info('=' * 80)
    logger.info(f'Config       : {args.config}')
    logger.info(f'Device       : {args.device}')
    logger.info(f'Physics      : {config.model.get("use_physics", True)}')
    logger.info(f'xTB method   : {config.model.get("xtb_method", "GFN2-xTB")}')
    logger.info(f'Gamma sched  : {config.model.get("gamma_schedule", "cosine")}')
    logger.info(f'Cache mode   : {"DISABLED (--no_cache)" if args.no_cache else "AUTO"}')
    logger.info('=' * 80)

    shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(config_path)))

    # ---- datasets ----
    logger.info('Loading datasets...')

    if args.no_cache:
        # Debug mode — plain dataset, online xTB
        transforms  = CountNodesPerGraph()
        train_set   = ConformationDataset(config.dataset.train, transform=transforms)
        val_set     = ConformationDataset(config.dataset.val,   transform=transforms)
        using_cache = False
        logger.info('Cache disabled. Using online xTB (slow).')
    else:
        train_set, train_cached = build_dataset(config, 'train', logger)
        val_set,   val_cached   = build_dataset(config, 'val',   logger)
        using_cache = train_cached

    logger.info(f'Training set size   : {len(train_set)}')
    logger.info(f'Validation set size : {len(val_set)}')
    logger.info(f'xTB cache active    : {using_cache}')

    # ---- dataloaders ----
    # Use custom collate only when cache is active (otherwise default is fine)
    collate_fn = collate_with_xtb if using_cache else None

    train_loader = DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    # ---- model ----
    logger.info('Building Hamiltonian Bridge model...')
    config.model.device = args.device
    model = HamiltonianBridgeNetwork(config.model).to(args.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Trainable parameters: {n_params:,}')

    # ---- optimizers ----
    optimizer_global  = get_optimizer(config.train.optimizer, model.model_global)
    optimizer_local   = get_optimizer(config.train.optimizer, model.model_local)
    scheduler_global  = get_scheduler(config.train.scheduler, optimizer_global)
    scheduler_local   = get_scheduler(config.train.scheduler, optimizer_local)

    start_iter = 1

    # ---- resume ----
    if resume:
        ckpt_path, start_iter = get_checkpoint_path(
            os.path.join(resume_from, 'checkpoints'), it=args.resume_iter
        )
        logger.info(f'Resuming from : {ckpt_path}')
        logger.info(f'Iteration     : {start_iter}')

        ckpt = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(ckpt['model'])
        optimizer_global.load_state_dict(ckpt['optimizer_global'])
        optimizer_local.load_state_dict(ckpt['optimizer_local'])
        scheduler_global.load_state_dict(ckpt['scheduler_global'])
        scheduler_local.load_state_dict(ckpt['scheduler_local'])

    # ---- training loop ----
    logger.info('Starting training...')
    train_iter = iter(train_loader)

    try:
        for it in range(start_iter, config.train.max_iters + 1):

            # infinite iterator over training data
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch      = next(train_iter)

            metrics = train_step(
                model, batch,
                optimizer_global, optimizer_local,
                config, args.device,
            )

            # ---- logging ----
            if it % config.train.log_freq == 0:
                logger.info(
                    f'[Train] Iter {it:05d} | '
                    f'Loss {metrics["loss"]:.4f} | '
                    f'Global {metrics["loss_global"]:.4f} | '
                    f'Local {metrics["loss_local"]:.4f} | '
                    f'Grad {metrics["grad_norm"]:.4f} | '
                    f'LR(G) {optimizer_global.param_groups[0]["lr"]:.6f} | '
                    f'LR(L) {optimizer_local.param_groups[0]["lr"]:.6f}'
                )
                writer.add_scalar('train/loss',        metrics['loss'],       it)
                writer.add_scalar('train/loss_global', metrics['loss_global'], it)
                writer.add_scalar('train/loss_local',  metrics['loss_local'],  it)
                writer.add_scalar('train/grad_norm',   metrics['grad_norm'],   it)
                writer.add_scalar('train/lr_global',   optimizer_global.param_groups[0]['lr'], it)
                writer.add_scalar('train/lr_local',    optimizer_local.param_groups[0]['lr'],  it)
                writer.flush()

            # ---- validation + checkpoint ----
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                val_metrics = validate(model, val_loader, config, args.device)

                logger.info(
                    f'[Val] Iter {it:05d} | '
                    f'Loss {val_metrics["loss"]:.6f} | '
                    f'Global {val_metrics["loss_global"]:.6f} | '
                    f'Local {val_metrics["loss_local"]:.6f}'
                )
                writer.add_scalar('val/loss',        val_metrics['loss'],        it)
                writer.add_scalar('val/loss_global', val_metrics['loss_global'], it)
                writer.add_scalar('val/loss_local',  val_metrics['loss_local'],  it)
                writer.flush()

                # schedulers
                if config.train.scheduler.type == 'plateau':
                    scheduler_global.step(val_metrics['loss_global'])
                    scheduler_local.step(val_metrics['loss_local'])
                else:
                    scheduler_global.step()
                    scheduler_local.step()

                # checkpoint
                ckpt_path = os.path.join(ckpt_dir, f'{it}.pt')
                torch.save({
                    'config':           config,
                    'model':            model.state_dict(),
                    'optimizer_global': optimizer_global.state_dict(),
                    'optimizer_local':  optimizer_local.state_dict(),
                    'scheduler_global': scheduler_global.state_dict(),
                    'scheduler_local':  scheduler_local.state_dict(),
                    'iteration':        it,
                    'val_loss':         val_metrics['loss'],
                }, ckpt_path)
                logger.info(f'Checkpoint saved: {ckpt_path}')

    except KeyboardInterrupt:
        logger.info('Training interrupted by user.')
    except Exception as e:
        logger.error(f'Training error: {e}')
        raise

    logger.info('Training completed!')