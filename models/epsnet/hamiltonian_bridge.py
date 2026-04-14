"""
Hamiltonian-Informed Schrödinger Bridge for Molecular Conformation Generation.

This module implements the proposal from the research presentation:
- Langevin reference SDE with xTB physics forces
- Warm-start initialization from xTB optimization
- Bridge network that learns xTB→DFT residual correction
- SE(3) equivariant throughout

Reference: "Physics-Aligned Geometric Diffusion for Molecular Conformation Generation"
"""

import torch
import torch.nn as nn
import numpy as np
from torch_scatter import scatter_add, scatter_mean
from tqdm.auto import tqdm

from ..common import MultiLayerPerceptron, assemble_atom_pair_feature, extend_graph_order_radius
from ..encoder import SchNetEncoder, GINEncoder, get_edge_encoder
from ..geometry import get_distance, eq_transform
from ..physics import DXTBForceField, GammaScheduler, WarmStartOptimizer


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """Get beta schedule for diffusion process."""
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "cosine":
        timesteps = np.arange(num_diffusion_timesteps + 1, dtype=np.float64) / num_diffusion_timesteps
        alphas = timesteps / (1 + timesteps)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, 0, 0.999)
    else:
        raise NotImplementedError(beta_schedule)

    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class HamiltonianBridgeNetwork(nn.Module):
    """
    Hamiltonian-Informed Schrödinger Bridge Network.

    Key innovations:
    1. Langevin reference SDE: dC_t = [-0.5*β(t)*C_t + γ(t)*f_xTB(C_t)]dt + sqrt(β(t))*dW_t
    2. Warm-start from xTB-optimized geometry
    3. Bridge learns residual: xTB → DFT level correction
    4. SE(3) equivariant architecture
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Edge encoders
        self.edge_encoder_global = get_edge_encoder(config)
        self.edge_encoder_local = get_edge_encoder(config)

        # Encoders (from GeoDiff)
        self.encoder_global = SchNetEncoder(
            hidden_channels=config.hidden_dim,
            num_filters=config.hidden_dim,
            num_interactions=config.num_convs,
            edge_channels=self.edge_encoder_global.out_channels,
            cutoff=config.cutoff,
            smooth=config.smooth_conv,
        )
        self.encoder_local = GINEncoder(
            hidden_dim=config.hidden_dim,
            num_convs=config.num_convs_local,
        )

        # Output MLPs
        self.grad_global_dist_mlp = MultiLayerPerceptron(
            2 * config.hidden_dim,
            [config.hidden_dim, config.hidden_dim // 2, 1],
            activation=config.mlp_act
        )
        self.grad_local_dist_mlp = MultiLayerPerceptron(
            2 * config.hidden_dim,
            [config.hidden_dim, config.hidden_dim // 2, 1],
            activation=config.mlp_act
        )

        # Model components
        self.model_global = nn.ModuleList([
            self.edge_encoder_global, self.encoder_global, self.grad_global_dist_mlp
        ])
        self.model_local = nn.ModuleList([
            self.edge_encoder_local, self.encoder_local, self.grad_local_dist_mlp
        ])

        # Diffusion parameters
        betas = get_beta_schedule(
            beta_schedule=config.beta_schedule,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            num_diffusion_timesteps=config.num_diffusion_timesteps,
        )
        betas = torch.from_numpy(betas).float()
        self.betas = nn.Parameter(betas, requires_grad=False)
        alphas = (1. - betas).cumprod(dim=0)
        self.alphas = nn.Parameter(alphas, requires_grad=False)
        self.num_timesteps = self.betas.size(0)

        # Physics components (NEW)
        self.use_physics = config.get('use_physics', True)
        if self.use_physics:
            self.physics_field = DXTBForceField(
                method=config.get('xtb_method', 'GFN2-xTB'),
                device=config.get('device', 'cpu')
            )
            self.gamma_scheduler = GammaScheduler(
                schedule_type=config.get('gamma_schedule', 'cosine'),
                num_timesteps=config.num_diffusion_timesteps,
                gamma_start=config.get('gamma_start', 1.0),
                gamma_end=config.get('gamma_end', 0.01)
            )
            self.warm_start_opt = WarmStartOptimizer(
                method=config.get('xtb_method', 'GFN2-xTB'),
                max_steps=config.get('warm_start_steps', 100)
            )
            # Weight for physics vs learned components
            self.physics_weight = config.get('physics_weight', 0.5)
        else:
            self.physics_field = None
            self.gamma_scheduler = None

    def forward(self, atom_type, pos, bond_index, bond_type, batch, time_step,
                edge_index=None, edge_type=None, edge_length=None, return_edges=False,
                extend_order=True, extend_radius=True, return_physics=False):
        """
        Forward pass: compute denoising score.

        Args:
            atom_type: (N,) atom types
            pos: (N, 3) coordinates
            bond_index: (2, E) bond indices
            bond_type: (E,) bond types
            batch: (N,) batch assignment
            time_step: (B,) timestep indices
            return_physics: if True, also return physics forces

        Returns:
            edge_inv_global: (E, 1) global edge scores
            edge_inv_local: (E_local, 1) local edge scores
            [physics_forces]: (N, 3) xTB forces (if return_physics=True)
        """
        N = atom_type.size(0)

        # Build edge graph if not provided
        if edge_index is None or edge_type is None or edge_length is None:
            edge_index, edge_type = extend_graph_order_radius(
                num_nodes=N,
                pos=pos,
                edge_index=bond_index,
                edge_type=bond_type,
                batch=batch,
                order=self.config.edge_order,
                cutoff=self.config.cutoff,
                extend_order=extend_order,
                extend_radius=extend_radius,
                is_sidechain=None,
            )
            edge_length = get_distance(pos, edge_index).unsqueeze(-1)

        local_edge_mask = is_local_edge(edge_type)

        # Learned components (standard GeoDiff architecture)
        sigma_edge = torch.ones(size=(edge_index.size(1), 1), device=pos.device)

        # Global encoding
        edge_attr_global = self.edge_encoder_global(
            edge_length=edge_length,
            edge_type=edge_type
        )
        node_attr_global = self.encoder_global(
            z=atom_type,
            edge_index=edge_index,
            edge_length=edge_length,
            edge_attr=edge_attr_global,
        )
        h_pair_global = assemble_atom_pair_feature(
            node_attr=node_attr_global,
            edge_index=edge_index,
            edge_attr=edge_attr_global,
        )
        edge_inv_global = self.grad_global_dist_mlp(h_pair_global) * (1.0 / sigma_edge)

        # Local encoding
        edge_attr_local = self.edge_encoder_local(
            edge_length=edge_length,
            edge_type=edge_type
        )
        node_attr_local = self.encoder_local(
            z=atom_type,
            edge_index=edge_index[:, local_edge_mask],
            edge_attr=edge_attr_local[local_edge_mask],
        )
        h_pair_local = assemble_atom_pair_feature(
            node_attr=node_attr_local,
            edge_index=edge_index[:, local_edge_mask],
            edge_attr=edge_attr_local[local_edge_mask],
        )
        edge_inv_local = self.grad_local_dist_mlp(h_pair_local) * (1.0 / sigma_edge[local_edge_mask])

        if return_edges:
            if return_physics and self.use_physics:
                # Compute physics forces
                _, physics_forces = self.physics_field(atom_type, pos, batch)
                return edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length, local_edge_mask, physics_forces
            else:
                return edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length, local_edge_mask
        else:
            return edge_inv_global, edge_inv_local

    def get_loss(self, atom_type, pos, bond_index, bond_type, batch, num_nodes_per_graph,
                 num_graphs, anneal_power=2.0, return_unreduced_loss=False):
        """
        Compute training loss for Hamiltonian Bridge.

        The loss trains the network to predict the residual correction
        from xTB-level to DFT-level structures.

        Args:
            atom_type: (N,) atom types
            pos: (N, 3) ground truth positions (DFT-level)
            bond_index: (2, E) bonds
            bond_type: (E,) bond types
            batch: (N,) batch assignment
            num_nodes_per_graph: number of nodes per graph
            num_graphs: number of graphs
            anneal_power: annealing power for loss weighting
            return_unreduced_loss: whether to return per-graph loss

        Returns:
            loss: total loss
            loss_global: global loss component
            loss_local: local loss component
        """
        N = atom_type.size(0)
        node2graph = batch

        # Sample timesteps (with symmetry trick from GeoDiff)
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=pos.device
        )
        time_step = torch.cat([time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]

        a = self.alphas.index_select(0, time_step)  # (G,)

        # Add noise to positions (forward diffusion)
        a_pos = a.index_select(0, node2graph).unsqueeze(-1)  # (N, 1)
        pos_noise = torch.zeros_like(pos)
        pos_noise.normal_()

        # Standard diffusion: C_t = sqrt(α_bar_t) * C_0 + sqrt(1 - α_bar_t) * ε
        pos_perturbed = pos * a_pos.sqrt() + pos_noise * (1.0 - a_pos).sqrt()

        # Physics-informed component (NEW)
        if self.use_physics and self.training:
            # Get γ(t) for physics weight
            gamma_t = self.gamma_scheduler(time_step)  # (G,)
            gamma_pos = gamma_t.index_select(0, node2graph).unsqueeze(-1)  # (N, 1)

            # Compute xTB forces on perturbed structure
            with torch.set_grad_enabled(True):
                pos_perturbed_grad = pos_perturbed.detach().requires_grad_(True)
                _, physics_forces = self.physics_field(atom_type, pos_perturbed_grad, batch)

            # Add physics drift to the perturbed positions
            # This implements the Langevin reference SDE
            pos_perturbed = pos_perturbed + gamma_pos * physics_forces * (1.0 - a_pos)

        # Forward pass through network
        edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length, local_edge_mask = self(
            atom_type=atom_type,
            pos=pos_perturbed,
            bond_index=bond_index,
            bond_type=bond_type,
            batch=batch,
            time_step=time_step,
            return_edges=True,
            extend_order=True,
            extend_radius=True
        )

        # Compute target (denoising direction)
        edge2graph = node2graph.index_select(0, edge_index[0])
        a_edge = a.index_select(0, edge2graph).unsqueeze(-1)

        d_gt = get_distance(pos, edge_index).unsqueeze(-1)
        d_perturbed = edge_length

        # Target: direction from perturbed to ground truth
        d_target = (d_gt - d_perturbed) / (1.0 - a_edge).sqrt() * a_edge.sqrt()

        # Global loss
        global_mask = torch.logical_and(
            torch.logical_or(d_perturbed <= self.config.cutoff, local_edge_mask.unsqueeze(-1)),
            ~local_edge_mask.unsqueeze(-1)
        )
        target_d_global = torch.where(global_mask, d_target, torch.zeros_like(d_target))
        edge_inv_global = torch.where(global_mask, edge_inv_global, torch.zeros_like(edge_inv_global))

        target_pos_global = eq_transform(target_d_global, pos_perturbed, edge_index, edge_length)
        node_eq_global = eq_transform(edge_inv_global, pos_perturbed, edge_index, edge_length)
        loss_global = (node_eq_global - target_pos_global) ** 2
        loss_global = 2 * torch.sum(loss_global, dim=-1, keepdim=True)

        # Local loss
        target_pos_local = eq_transform(
            d_target[local_edge_mask], pos_perturbed,
            edge_index[:, local_edge_mask], edge_length[local_edge_mask]
        )
        node_eq_local = eq_transform(
            edge_inv_local, pos_perturbed,
            edge_index[:, local_edge_mask], edge_length[local_edge_mask]
        )
        loss_local = (node_eq_local - target_pos_local) ** 2
        loss_local = 5 * torch.sum(loss_local, dim=-1, keepdim=True)

        # Total loss
        loss = loss_global + loss_local

        if return_unreduced_loss:
            return loss, loss_global, loss_local
        else:
            return loss.mean(), loss_global.mean(), loss_local.mean()

    def sample(self, atom_type, bond_index, bond_type, batch, num_graphs,
               n_steps=None, use_warm_start=True, device='cpu'):
        """
        Sample conformations using Hamiltonian-Informed Schrödinger Bridge.

        Pipeline:
        1. (Optional) Warm-start from xTB optimization
        2. Langevin sampling with physics-informed reference SDE
        3. Bridge network learns residual correction

        Args:
            atom_type: (N,) atom types
            bond_index: (2, E) bonds
            bond_type: (E,) bond types
            batch: (N,) batch assignment
            num_graphs: number of molecules
            n_steps: number of sampling steps (default: all timesteps)
            use_warm_start: whether to use xTB warm-start
            device: torch device

        Returns:
            pos_final: (N, 3) generated coordinates
            pos_traj: list of intermediate positions
        """
        N = atom_type.size(0)

        if n_steps is None:
            n_steps = self.num_timesteps

        # Step 1: Warm-start initialization (NEW)
        if use_warm_start and self.use_physics:
            print("Initializing with xTB warm-start...")
            # Random initialization
            pos_init = torch.randn(N, 3, device=device)
            pos_init = center_pos(pos_init, batch)

            # Optimize with xTB
            pos_warm, _ = self.warm_start_opt.optimize(
                atom_type, pos_init, batch, device
            )
            # Add noise scaled by final sigma
            sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt()
            pos = pos_warm + torch.randn_like(pos_warm) * sigmas[-1] * 0.5
        else:
            # Standard noise initialization
            sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt()
            pos = torch.randn(N, 3, device=device) * sigmas[-1]

        pos = center_pos(pos, batch)

        # Step 2: Langevin sampling with physics-informed SDE
        pos_traj = []

        with torch.no_grad():
            seq = range(self.num_timesteps - n_steps, self.num_timesteps)
            seq_next = [-1] + list(seq[:-1])

            for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), desc='Sampling', total=len(seq)):
                t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=device)

                # Get learned score
                edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length, local_edge_mask = self(
                    atom_type=atom_type,
                    pos=pos,
                    bond_index=bond_index,
                    bond_type=bond_type,
                    batch=batch,
                    time_step=t,
                    return_edges=True,
                    extend_order=True,
                    extend_radius=True
                )

                # Compute learned denoising direction
                node_eq_local = eq_transform(
                    edge_inv_local, pos, edge_index[:, local_edge_mask], edge_length[local_edge_mask]
                )
                edge_inv_global = edge_inv_global * (1 - local_edge_mask.view(-1, 1).float())
                node_eq_global = eq_transform(edge_inv_global, pos, edge_index, edge_length)

                eps_learned = node_eq_local + node_eq_global * 0.2

                # Add physics component (NEW - Langevin SDE)
                if self.use_physics:
                    gamma_t = self.gamma_scheduler(t)  # (G,)
                    gamma_pos = gamma_t.index_select(0, batch).unsqueeze(-1)  # (N, 1)

                    # Compute xTB forces
                    _, physics_forces = self.physics_field(atom_type, pos.detach(), batch)

                    # Combined: learned + physics
                    eps_total = eps_learned + gamma_pos * physics_forces * self.physics_weight
                else:
                    eps_total = eps_learned

                # DDPM update step
                b = self.betas
                t_idx = t[0]
                next_t = torch.tensor([j], device=device)

                at = compute_alpha(b, t_idx.long())
                at_next = compute_alpha(b, next_t.long())

                beta_t = 1 - at / at_next
                e = -eps_total
                pos0_from_e = (1.0 / at).sqrt() * pos - (1.0 / at - 1).sqrt() * e
                mean = ((at_next.sqrt() * beta_t) * pos0_from_e + ((1 - beta_t).sqrt() * (1 - at_next)) * pos) / (1.0 - at)

                noise = torch.randn_like(pos)
                mask = 1 - (t_idx == 0).float()
                logvar = beta_t.log()
                pos_next = mean + mask * torch.exp(0.5 * logvar) * noise

                pos = center_pos(pos_next, batch)
                pos_traj.append(pos.clone().cpu())

        return pos, pos_traj


def compute_alpha(beta, t):
    """Compute cumulative product of (1 - beta)."""
    beta = torch.cat([torch.zeros(1, device=beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1)
    return a


def center_pos(pos, batch):
    """Center positions to zero center of mass."""
    pos_center = pos - scatter_mean(pos, batch, dim=0)[batch]
    return pos_center


def is_local_edge(edge_type):
    """Check if edge is a local (bond) edge."""
    return edge_type > 0
