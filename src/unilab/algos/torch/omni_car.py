from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from tensordict import TensorDict


class OmniCarGridCNNModel(MLPModel):
    """CNN encoder for OmniCar occupancy grids plus an MLP low-state head."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        grid_size: int = 80,
        cnn_feature_dim: int = 128,
    ) -> None:
        self.grid_size = int(grid_size)
        self.grid_dim = self.grid_size * self.grid_size
        self.cnn_feature_dim = int(cnn_feature_dim)
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )
        if self.obs_dim <= self.grid_dim:
            raise ValueError(
                f"OmniCarGridCNNModel expects grid plus low-state inputs; got obs_dim={self.obs_dim}"
            )
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(32 * 10 * 10, self.cnn_feature_dim),
            nn.ELU(),
        )

    def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
        del masks, hidden_state
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        flat = torch.cat(obs_list, dim=-1)
        flat = self.obs_normalizer(flat)
        grid = flat[..., : self.grid_dim].reshape(-1, 1, self.grid_size, self.grid_size)
        low_state = flat[..., self.grid_dim :]
        grid_features = self.grid_encoder(grid)
        return torch.cat([grid_features, low_state], dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            obs_list = [obs[obs_group] for obs_group in self.obs_groups]
            self.obs_normalizer.update(torch.cat(obs_list, dim=-1))  # type: ignore[attr-defined]

    def _get_latent_dim(self) -> int:
        low_state_dim = self.obs_dim - self.grid_dim
        return self.cnn_feature_dim + low_state_dim
