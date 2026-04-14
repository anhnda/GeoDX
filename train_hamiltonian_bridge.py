"""
Training script for Hamiltonian-Informed Schrödinger Bridge.

This implements the physics-aligned geometric diffusion model
with xTB-informed Langevin dynamics.
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
from torch_geometric.data import DataLoader

from models.epsnet.hamiltonian_bridge import HamiltonianBridgeNetwork
from utils.datasets import ConformationDataset
from utils.transforms import *
from utils.misc import *
from utils.common import get_optimizer, get_scheduler


def train_step(model, batch, optimizer_global, optimizer_local, config, device):
    """Single training step."""
    model.train()
    optimizer_global.zero_grad()
    optimizer_local.zero_grad()

    batch = batch.to(device)

    # Compute loss
    loss, loss_global, loss_local = model.get_loss(
        atom_type=batch.atom_type,
        pos=batch.pos,
        bond_index=batch.edge_index,
        bond_type=batch.edge_type,
        batch=batch.batch,
        num_nodes_per_graph=batch.num_nodes_per_graph,
        num_graphs=batch.num_graphs,
        anneal_power=config.train.anneal_power,
        return_unreduced_loss=False
    )

    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
    optimizer_global.step()
    optimizer_local.step()

    return {
        'loss': loss.item(),
        'loss_global': loss_global.item(),
        'loss_local': loss_local.item(),
        'grad_norm': orig_grad_norm
    }


def validate(model, val_loader, config, device):
    """Validation loop."""
    model.eval()
    sum_loss = 0.0
    sum_loss_global = 0.0
    sum_loss_local = 0.0
    count = 0

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
                return_unreduced_loss=False
            )

            sum_loss += loss.item() * batch.num_graphs
            sum_loss_global += loss_global.item() * batch.num_graphs
            sum_loss_local += loss_local.item() * batch.num_graphs
            count += batch.num_graphs

    return {
        'loss': sum_loss / count,
        'loss_global': sum_loss_global / count,
        'loss_local': sum_loss_local / count
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume_iter', type=int, default=None)
    parser.add_argument('--logdir', type=str, default='./logs_hamiltonian')
    args = parser.parse_args()

    # Load config
    resume = os.path.isdir(args.config)
    if resume:
        config_path = glob(os.path.join(args.config, '*.yml'))[0]
        resume_from = args.config
    else:
        config_path = args.config

    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]
    seed_all(config.train.seed)

    # Setup logging
    if resume:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag='resume')
        os.symlink(os.path.realpath(resume_from), os.path.join(log_dir, os.path.basename(resume_from.rstrip("/"))))
    else:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name)
        shutil.copytree('./models', os.path.join(log_dir, 'models'))

    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = get_logger('train_hamiltonian', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)

    logger.info("=" * 80)
    logger.info("Hamiltonian-Informed Schrödinger Bridge Training")
    logger.info("=" * 80)
    logger.info(f"Config: {args}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Physics enabled: {config.model.get('use_physics', True)}")
    logger.info(f"xTB method: {config.model.get('xtb_method', 'GFN2-xTB')}")
    logger.info(f"Gamma schedule: {config.model.get('gamma_schedule', 'cosine')}")
    logger.info("=" * 80)

    shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(config_path)))

    # Load datasets
    logger.info('Loading datasets...')
    transforms = CountNodesPerGraph()
    train_set = ConformationDataset(config.dataset.train, transform=transforms)
    val_set = ConformationDataset(config.dataset.val, transform=transforms)

    train_loader = DataLoader(
        train_set, batch_size=config.train.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_set, batch_size=config.train.batch_size, shuffle=False, num_workers=4
    )

    logger.info(f'Training set size: {len(train_set)}')
    logger.info(f'Validation set size: {len(val_set)}')

    # Build model
    logger.info('Building Hamiltonian Bridge model...')
    config.model.device = args.device  # Add device to config for physics module
    model = HamiltonianBridgeNetwork(config.model).to(args.device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Total trainable parameters: {n_params:,}')

    # Optimizers
    optimizer_global = get_optimizer(config.train.optimizer, model.model_global)
    optimizer_local = get_optimizer(config.train.optimizer, model.model_local)
    scheduler_global = get_scheduler(config.train.scheduler, optimizer_global)
    scheduler_local = get_scheduler(config.train.scheduler, optimizer_local)

    start_iter = 1

    # Resume from checkpoint
    if resume:
        ckpt_path, start_iter = get_checkpoint_path(
            os.path.join(resume_from, 'checkpoints'), it=args.resume_iter
        )
        logger.info(f'Resuming from: {ckpt_path}')
        logger.info(f'Iteration: {start_iter}')

        ckpt = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(ckpt['model'])
        optimizer_global.load_state_dict(ckpt['optimizer_global'])
        optimizer_local.load_state_dict(ckpt['optimizer_local'])
        scheduler_global.load_state_dict(ckpt['scheduler_global'])
        scheduler_local.load_state_dict(ckpt['scheduler_local'])

    # Training loop
    logger.info('Starting training...')
    train_iter = iter(train_loader)

    try:
        for it in range(start_iter, config.train.max_iters + 1):
            # Get next batch (with infinite iterator)
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # Train step
            metrics = train_step(
                model, batch, optimizer_global, optimizer_local, config, args.device
            )

            # Log
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

                writer.add_scalar('train/loss', metrics['loss'], it)
                writer.add_scalar('train/loss_global', metrics['loss_global'], it)
                writer.add_scalar('train/loss_local', metrics['loss_local'], it)
                writer.add_scalar('train/grad_norm', metrics['grad_norm'], it)
                writer.add_scalar('train/lr_global', optimizer_global.param_groups[0]['lr'], it)
                writer.add_scalar('train/lr_local', optimizer_local.param_groups[0]['lr'], it)
                writer.flush()

            # Validation
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                val_metrics = validate(model, val_loader, config, args.device)

                logger.info(
                    f'[Val] Iter {it:05d} | '
                    f'Loss {val_metrics["loss"]:.6f} | '
                    f'Global {val_metrics["loss_global"]:.6f} | '
                    f'Local {val_metrics["loss_local"]:.6f}'
                )

                writer.add_scalar('val/loss', val_metrics['loss'], it)
                writer.add_scalar('val/loss_global', val_metrics['loss_global'], it)
                writer.add_scalar('val/loss_local', val_metrics['loss_local'], it)
                writer.flush()

                # Update schedulers
                if config.train.scheduler.type == 'plateau':
                    scheduler_global.step(val_metrics['loss_global'])
                    scheduler_local.step(val_metrics['loss_local'])
                else:
                    scheduler_global.step()
                    scheduler_local.step()

                # Save checkpoint
                ckpt_path = os.path.join(ckpt_dir, f'{it}.pt')
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer_global': optimizer_global.state_dict(),
                    'optimizer_local': optimizer_local.state_dict(),
                    'scheduler_global': scheduler_global.state_dict(),
                    'scheduler_local': scheduler_local.state_dict(),
                    'iteration': it,
                    'val_loss': val_metrics['loss'],
                }, ckpt_path)
                logger.info(f'Checkpoint saved: {ckpt_path}')

    except KeyboardInterrupt:
        logger.info('Training interrupted by user')
    except Exception as e:
        logger.error(f'Training error: {e}')
        raise

    logger.info('Training completed!')
