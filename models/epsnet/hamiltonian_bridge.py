"""
Hamiltonian-Informed Schrödinger Bridge for Molecular Conformation Generation.

Changes vs. original
--------------------
get_loss() now accepts an optional `precomputed_forces` argument.
When provided (from CachedXTBDataset), xTB is NOT called during training —
forces are loaded from disk instead, giving ~100x speedup per batch.
When None, falls back to online xTB (original behaviour).
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
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == 'sigmoid':
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif beta_schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'cosine':
        timesteps = np.arange(num_diffusion_timesteps + 1, dtype=np.float64) / num_diffusion_timesteps
        alphas    = timesteps / (1 + timesteps)
        alphas    = alphas / alphas[0]
        betas     = 1 - alphas[1:] / alphas[:-1]
        betas     = np.clip(betas, 0, 0.999)
    else:
        raise NotImplementedError(beta_schedule)

    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class HamiltonianBridgeNetwork(nn.Module):
    """
    Hamiltonian-Informed Schrödinger Bridge Network.

    Key innovations:
    1. Langevin reference SDE:
         dC_t = [-0.5*β(t)*C_t + γ(t)*f_xTB(C_t)] dt + sqrt(β(t))*dW_t
    2. Warm-start initialization from xTB-optimized geometry
    3. Bridge learns residual: xTB → DFT level correction
    4. SE(3) equivariant throughout
    5. (NEW) Precomputed xTB forces loaded from disk — no online QM call during training
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Edge encoders
        self.edge_encoder_global = get_edge_encoder(config)
        self.edge_encoder_local  = get_edge_encoder(config)

        # Encoders
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
            activation=config.mlp_act,
        )
        self.grad_local_dist_mlp = MultiLayerPerceptron(
            2 * config.hidden_dim,
            [config.hidden_dim, config.hidden_dim // 2, 1],
            activation=config.mlp_act,
        )

        # Module lists (for separate optimizers in train.py)
        self.model_global = nn.ModuleList([
            self.edge_encoder_global, self.encoder_global, self.grad_global_dist_mlp
        ])
        self.model_local = nn.ModuleList([
            self.edge_encoder_local, self.encoder_local, self.grad_local_dist_mlp
        ])

        # Diffusion parameters
        betas  = get_beta_schedule(
            beta_schedule=config.beta_schedule,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            num_diffusion_timesteps=config.num_diffusion_timesteps,
        )
        betas  = torch.from_numpy(betas).float()
        self.betas  = nn.Parameter(betas, requires_grad=False)
        alphas = (1. - betas).cumprod(dim=0)
        self.alphas = nn.Parameter(alphas, requires_grad=False)
        self.num_timesteps = self.betas.size(0)

        # Physics components
        self.use_physics = config.get('use_physics', True)
        if self.use_physics:
            self.physics_field = DXTBForceField(
                method=config.get('xtb_method', 'GFN2-xTB'),
                device=config.get('device', 'cpu'),
            )
            self.gamma_scheduler = GammaScheduler(
                schedule_type=config.get('gamma_schedule', 'cosine'),
                num_timesteps=config.num_diffusion_timesteps,
                gamma_start=config.get('gamma_start', 1.0),
                gamma_end=config.get('gamma_end', 0.01),
            )
            self.warm_start_opt = WarmStartOptimizer(
                method=config.get('xtb_method', 'GFN2-xTB'),
                max_steps=config.get('warm_start_steps', 100),
            )
            self.physics_weight = config.get('physics_weight', 0.5)
        else:
            self.physics_field   = None
            self.gamma_scheduler = None

    # ------------------------------------------------------------------
    def forward(self, atom_type, pos, bond_index, bond_type, batch, time_step,
                edge_index=None, edge_type=None, edge_length=None,
                return_edges=False, extend_order=True, extend_radius=True,
                return_physics=False):
        """Compute denoising score (unchanged from original)."""
        N = atom_type.size(0)

        if edge_index is None or edge_type is None or edge_length is None:
            edge_index, edge_type = extend_graph_order_radius(
                num_nodes=N, pos=pos,
                edge_index=bond_index, edge_type=bond_type,
                batch=batch, order=self.config.edge_order,
                cutoff=self.config.cutoff,
                extend_order=extend_order, extend_radius=extend_radius,
                is_sidechain=None,
            )
            edge_length = get_distance(pos, edge_index).unsqueeze(-1)

        local_edge_mask = is_local_edge(edge_type)
        sigma_edge      = torch.ones(size=(edge_index.size(1), 1), device=pos.device)

        # Global branch
        edge_attr_global = self.edge_encoder_global(
            edge_length=edge_length, edge_type=edge_type
        )
        node_attr_global = self.encoder_global(
            z=atom_type, edge_index=edge_index,
            edge_length=edge_length, edge_attr=edge_attr_global,
        )
        h_pair_global   = assemble_atom_pair_feature(
            node_attr=node_attr_global, edge_index=edge_index,
            edge_attr=edge_attr_global,
        )
        edge_inv_global = self.grad_global_dist_mlp(h_pair_global) * (1.0 / sigma_edge)

        # Local branch
        edge_attr_local = self.edge_encoder_local(
            edge_length=edge_length, edge_type=edge_type
        )
        node_attr_local = self.encoder_local(
            z=atom_type,
            edge_index=edge_index[:, local_edge_mask],
            edge_attr=edge_attr_local[local_edge_mask],
        )
        h_pair_local   = assemble_atom_pair_feature(
            node_attr=node_attr_local,
            edge_index=edge_index[:, local_edge_mask],
            edge_attr=edge_attr_local[local_edge_mask],
        )
        edge_inv_local = self.grad_local_dist_mlp(h_pair_local) * (
            1.0 / sigma_edge[local_edge_mask]
        )

        if return_edges:
            if return_physics and self.use_physics:
                with torch.no_grad():
                    _, physics_forces = self.physics_field(atom_type, pos, batch)
                return (edge_inv_global, edge_inv_local,
                        edge_index, edge_type, edge_length, local_edge_mask,
                        physics_forces)
            return (edge_inv_global, edge_inv_local,
                    edge_index, edge_type, edge_length, local_edge_mask)

        return edge_inv_global, edge_inv_local

    # ------------------------------------------------------------------
    def _get_physics_forces(self, atom_type, pos, batch, precomputed_forces=None):
        """
        Return xTB forces, using precomputed cache when available.

        Priority:
            1. precomputed_forces  — loaded from disk, free
            2. online xTB          — slow, fallback only

        Forces are clamped to [-1, 1] and detached in both paths so
        gradients never flow through the physics oracle.

        Parameters
        ----------
        atom_type          : (N,) atom type indices
        pos                : (N, 3) current positions  [only used for online path]
        batch              : (N,)  batch assignment    [only used for online path]
        precomputed_forces : (N, 3) | None

        Returns
        -------
        forces : (N, 3) detached, clamped
        """
        if precomputed_forces is not None:
            # Fast path: use precomputed forces from CachedXTBDataset
            forces = precomputed_forces.to(pos.device)
        else:
            # Slow path: call xTB online (original behaviour)
            with torch.no_grad():
                _, forces = self.physics_field(atom_type, pos, batch)

        return torch.clamp(forces, min=-1.0, max=1.0).detach()

    # ------------------------------------------------------------------
    def get_loss(self, atom_type, pos, bond_index, bond_type, batch,
                 num_nodes_per_graph, num_graphs,
                 anneal_power=2.0, return_unreduced_loss=False,
                 precomputed_forces=None):
        """
        Compute training loss for Hamiltonian Bridge.

        Parameters
        ----------
        atom_type           : (N,)   atom type indices
        pos                 : (N, 3) ground-truth DFT positions
        bond_index          : (2, E) bond edge indices
        bond_type           : (E,)   bond types
        batch               : (N,)   batch assignment
        num_nodes_per_graph : (B,)   nodes per graph
        num_graphs          : int    batch size
        anneal_power        : float  loss annealing exponent
        return_unreduced_loss : bool return per-atom tensors
        precomputed_forces  : (N, 3) | None
            Pre-loaded xTB forces from CachedXTBDataset.
            When provided, no xTB call is made → ~100x faster training.
            When None, falls back to online xTB (original behaviour).

        Returns
        -------
        loss, loss_global, loss_local  — scalars (or per-atom if return_unreduced_loss)
        """
        N         = atom_type.size(0)
        node2graph = batch

        # ---- sample timesteps (symmetry trick from GeoDiff) ----
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=pos.device
        )
        time_step = torch.cat(
            [time_step, self.num_timesteps - time_step - 1], dim=0
        )[:num_graphs]

        a     = self.alphas.index_select(0, time_step)        # (G,)
        a_pos = a.index_select(0, node2graph).unsqueeze(-1)   # (N, 1)

        # ---- forward diffusion ----
        pos_noise    = torch.zeros_like(pos).normal_()
        pos_perturbed = pos * a_pos.sqrt() + pos_noise * (1.0 - a_pos).sqrt()

        # ---- physics-informed Langevin drift (NEW: uses cache when available) ----
        if self.use_physics and self.training:
            gamma_t   = self.gamma_scheduler(time_step)                  # (G,)
            gamma_pos = gamma_t.index_select(0, node2graph).unsqueeze(-1) # (N, 1)

            # *** KEY CHANGE ***
            # precomputed_forces → free disk load (~0 cost)
            # None               → online xTB call (expensive, fallback)
            physics_forces = self._get_physics_forces(
                atom_type, pos_perturbed, node2graph,
                precomputed_forces=precomputed_forces,
            )

            # Langevin reference SDE drift: push perturbed pos toward xTB basin
            pos_perturbed = pos_perturbed + gamma_pos * physics_forces * (1.0 - a_pos)

        # ---- forward pass through network ----
        (edge_inv_global, edge_inv_local,
         edge_index, edge_type, edge_length,
         local_edge_mask) = self(
            atom_type=atom_type,
            pos=pos_perturbed,
            bond_index=bond_index,
            bond_type=bond_type,
            batch=batch,
            time_step=time_step,
            return_edges=True,
            extend_order=True,
            extend_radius=True,
        )

        # ---- denoising targets ----
        edge2graph = node2graph.index_select(0, edge_index[0])
        a_edge     = a.index_select(0, edge2graph).unsqueeze(-1)

        d_gt        = get_distance(pos,           edge_index).unsqueeze(-1)
        d_perturbed = edge_length
        d_target    = (d_gt - d_perturbed) / ((1.0 - a_edge).sqrt() + 1e-8) * a_edge.sqrt()

        # ---- global loss ----
        global_mask = torch.logical_and(
            torch.logical_or(
                d_perturbed <= self.config.cutoff,
                local_edge_mask.unsqueeze(-1)
            ),
            ~local_edge_mask.unsqueeze(-1),
        )
        target_d_global  = torch.where(global_mask, d_target,       torch.zeros_like(d_target))
        edge_inv_global  = torch.where(global_mask, edge_inv_global, torch.zeros_like(edge_inv_global))
        target_pos_global = eq_transform(target_d_global, pos_perturbed, edge_index, edge_length)
        node_eq_global    = eq_transform(edge_inv_global, pos_perturbed, edge_index, edge_length)
        loss_global       = 2 * torch.sum((node_eq_global - target_pos_global) ** 2,
                                          dim=-1, keepdim=True)

        # ---- local loss ----
        target_pos_local = eq_transform(
            d_target[local_edge_mask], pos_perturbed,
            edge_index[:, local_edge_mask], edge_length[local_edge_mask],
        )
        node_eq_local = eq_transform(
            edge_inv_local, pos_perturbed,
            edge_index[:, local_edge_mask], edge_length[local_edge_mask],
        )
        loss_local = 5 * torch.sum((node_eq_local - target_pos_local) ** 2,
                                   dim=-1, keepdim=True)

        loss = loss_global + loss_local

        if return_unreduced_loss:
            return loss, loss_global, loss_local
        return loss.mean(), loss_global.mean(), loss_local.mean()

    # ------------------------------------------------------------------
    def sample(self, atom_type, bond_index, bond_type, batch, num_graphs,
               n_steps=None, use_warm_start=True, device='cpu'):
        """
        Sample conformations using Hamiltonian-Informed Schrödinger Bridge.

        Pipeline:
        1. (Optional) Warm-start from xTB optimisation
        2. Langevin reverse SDE with physics-informed reference
        3. Bridge network learns residual correction

        Note: sampling always calls xTB online (precomputed cache is training-only).
        """
        N = atom_type.size(0)
        if n_steps is None:
            n_steps = self.num_timesteps

        # ---- initialisation ----
        if use_warm_start and self.use_physics:
            print('Initializing with xTB warm-start...')
            pos_init = center_pos(torch.randn(N, 3, device=device), batch)
            pos_warm, _ = self.warm_start_opt.optimize(atom_type, pos_init, batch, device)
            sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt()
            pos = pos_warm + torch.randn_like(pos_warm) * sigmas[-1] * 0.5
        else:
            sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt()
            pos    = torch.randn(N, 3, device=device) * sigmas[-1]

        pos = center_pos(pos, batch)
        pos_traj = []

        with torch.no_grad():
            seq      = range(self.num_timesteps - n_steps, self.num_timesteps)
            seq_next = [-1] + list(seq[:-1])

            for i, j in tqdm(
                zip(reversed(seq), reversed(seq_next)),
                desc='Sampling', total=len(seq)
            ):
                t = torch.full((num_graphs,), i, dtype=torch.long, device=device)

                (edge_inv_global, edge_inv_local,
                 edge_index, edge_type, edge_length,
                 local_edge_mask) = self(
                    atom_type=atom_type, pos=pos,
                    bond_index=bond_index, bond_type=bond_type,
                    batch=batch, time_step=t,
                    return_edges=True,
                    extend_order=True, extend_radius=True,
                )

                # learned denoising direction
                node_eq_local   = eq_transform(
                    edge_inv_local, pos,
                    edge_index[:, local_edge_mask], edge_length[local_edge_mask],
                )
                edge_inv_global = edge_inv_global * (
                    1 - local_edge_mask.view(-1, 1).float()
                )
                node_eq_global  = eq_transform(edge_inv_global, pos, edge_index, edge_length)
                eps_learned     = node_eq_local + node_eq_global * 0.2

                # physics component (online xTB during sampling — no cache needed)
                if self.use_physics:
                    gamma_t   = self.gamma_scheduler(t)
                    gamma_pos = gamma_t.index_select(0, batch).unsqueeze(-1)
                    with torch.enable_grad():
                        _, physics_forces = self.physics_field(atom_type, pos, batch)
                    physics_forces = torch.clamp(physics_forces, -1.0, 1.0)
                    eps_total = eps_learned + gamma_pos * physics_forces * self.physics_weight
                else:
                    eps_total = eps_learned

                # DDPM update
                b        = self.betas
                t_idx    = t[0]
                next_t   = torch.tensor([j], device=device)
                at       = compute_alpha(b, t_idx.long())
                at_next  = compute_alpha(b, next_t.long())
                beta_t   = 1 - at / at_next

                e            = -eps_total
                pos0_from_e  = (1.0 / at).sqrt() * pos - (1.0 / at - 1).sqrt() * e
                mean         = (
                    (at_next.sqrt() * beta_t) * pos0_from_e
                    + ((1 - beta_t).sqrt() * (1 - at_next)) * pos
                ) / (1.0 - at)

                noise   = torch.randn_like(pos)
                mask    = 1 - (t_idx == 0).float()
                logvar  = beta_t.log()
                pos_next = mean + mask * torch.exp(0.5 * logvar) * noise

                pos = center_pos(pos_next, batch)
                pos_traj.append(pos.clone().cpu())

        return pos, pos_traj


# ============================================================
#  Utilities
# ============================================================

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1, device=beta.device), beta], dim=0)
    return (1 - beta).cumprod(dim=0).index_select(0, t + 1)


def center_pos(pos, batch):
    return pos - scatter_mean(pos, batch, dim=0)[batch]


def is_local_edge(edge_type):
    return edge_type > 0