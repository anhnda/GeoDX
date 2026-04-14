"""
Gamma(t) scheduling for physics drift weight in Langevin SDE.

The physics drift weight γ(t) controls the contribution of xTB forces
at different diffusion timesteps. Typically, γ(t) should be:
  - Large at early timesteps (high noise) to guide the process
  - Small near t=0 to allow fine-tuning by the learned model
"""

import numpy as np
import torch
import torch.nn as nn


class GammaScheduler(nn.Module):
    """
    Scheduler for γ(t) - the physics drift weight coefficient.
    """

    def __init__(self, schedule_type='cosine', num_timesteps=5000,
                 gamma_start=1.0, gamma_end=0.01):
        """
        Args:
            schedule_type: 'constant', 'linear', 'cosine', 'exponential'
            num_timesteps: number of diffusion timesteps
            gamma_start: γ value at t=T (start of reverse process)
            gamma_end: γ value at t=0 (end of reverse process)
        """
        super().__init__()
        self.schedule_type = schedule_type
        self.num_timesteps = num_timesteps
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end

        # Precompute gamma schedule
        gammas = get_gamma_schedule(
            schedule_type=schedule_type,
            num_timesteps=num_timesteps,
            gamma_start=gamma_start,
            gamma_end=gamma_end
        )
        self.register_buffer('gammas', torch.from_numpy(gammas).float())

    def forward(self, t):
        """
        Get γ(t) for given timesteps.

        Args:
            t: (B,) timestep indices

        Returns:
            gamma: (B,) gamma values
        """
        return self.gammas[t]

    def get_gamma_at_timestep(self, t):
        """Alias for forward()."""
        return self.forward(t)


def get_gamma_schedule(schedule_type, num_timesteps, gamma_start=1.0, gamma_end=0.01):
    """
    Generate γ(t) schedule.

    Args:
        schedule_type: type of schedule
        num_timesteps: number of timesteps
        gamma_start: starting value
        gamma_end: ending value

    Returns:
        gammas: (num_timesteps,) array of gamma values
    """
    t = np.arange(num_timesteps, dtype=np.float64)

    if schedule_type == 'constant':
        # Constant γ throughout
        gammas = np.ones(num_timesteps) * gamma_start

    elif schedule_type == 'linear':
        # Linear decay from gamma_start to gamma_end
        gammas = np.linspace(gamma_start, gamma_end, num_timesteps)

    elif schedule_type == 'cosine':
        # Cosine decay (smooth)
        # γ(t) = gamma_end + (gamma_start - gamma_end) * 0.5 * (1 + cos(πt/T))
        gammas = gamma_end + (gamma_start - gamma_end) * 0.5 * (
            1 + np.cos(np.pi * t / num_timesteps)
        )

    elif schedule_type == 'exponential':
        # Exponential decay
        # γ(t) = gamma_start * (gamma_end/gamma_start)^(t/T)
        ratio = gamma_end / gamma_start
        gammas = gamma_start * (ratio ** (t / num_timesteps))

    elif schedule_type == 'inverse_sqrt':
        # Inverse square root: γ(t) ∝ 1/sqrt(T-t+1)
        # Stronger physics at high noise levels
        remaining = num_timesteps - t + 1
        gammas_norm = 1.0 / np.sqrt(remaining)
        # Normalize to [gamma_end, gamma_start]
        gammas_norm = (gammas_norm - gammas_norm.min()) / (gammas_norm.max() - gammas_norm.min())
        gammas = gamma_end + (gamma_start - gamma_end) * gammas_norm

    elif schedule_type == 'sigmoid':
        # Sigmoid-based schedule for smooth transition
        # γ(t) = gamma_end + (gamma_start - gamma_end) / (1 + exp(k*(t/T - 0.5)))
        k = 10  # steepness parameter
        t_norm = t / num_timesteps
        sigmoid_vals = 1.0 / (1.0 + np.exp(k * (t_norm - 0.5)))
        gammas = gamma_end + (gamma_start - gamma_end) * sigmoid_vals

    else:
        raise NotImplementedError(f"Unknown schedule type: {schedule_type}")

    return gammas


class AdaptiveGammaScheduler(GammaScheduler):
    """
    Adaptive γ(t) scheduler that adjusts based on denoising progress.
    Can modify γ based on current noise level or prediction quality.
    """

    def __init__(self, base_schedule='cosine', num_timesteps=5000,
                 gamma_start=1.0, gamma_end=0.01, adaptive_factor=0.5):
        """
        Args:
            base_schedule: base schedule type
            adaptive_factor: how much to adapt (0=no adaptation, 1=full adaptation)
        """
        super().__init__(base_schedule, num_timesteps, gamma_start, gamma_end)
        self.adaptive_factor = adaptive_factor
        self.register_buffer('gamma_adjustments', torch.ones(num_timesteps))

    def adapt(self, t, prediction_error):
        """
        Adapt γ based on prediction quality.

        Args:
            t: timestep index
            prediction_error: measure of how well the model is predicting
        """
        # Higher error -> increase physics contribution
        adjustment = 1.0 + self.adaptive_factor * prediction_error
        self.gamma_adjustments[t] = adjustment.item()

    def forward(self, t):
        """Get adapted γ(t)."""
        base_gamma = self.gammas[t]
        adjustment = self.gamma_adjustments[t]
        return base_gamma * adjustment
