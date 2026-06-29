from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from rsl_rl.utils import resolve_nn_activation
from tensordict import TensorDict


def _normalize_activation_name(activation: str) -> str:
    normalized = activation.lower()
    return "swish" if normalized == "silu" else normalized


def _activation(activation: str) -> nn.Module:
    return resolve_nn_activation(_normalize_activation_name(activation))


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
        activation = _normalize_activation_name(activation)
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
            _activation(activation),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            _activation(activation),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            _activation(activation),
            nn.Flatten(),
            nn.Linear(32 * 10 * 10, self.cnn_feature_dim),
            _activation(activation),
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


class OmniCarGridCNNGRUModel(MLPModel):
    """CNN encoder over grid frames, GRU temporal fusion, then an MLP policy head."""

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
        grid_history_len: int = 10,
        grid_cell_size: float = 0.05,
        cnn_feature_dim: int = 128,
        gru_hidden_dim: int = 128,
        gru_layers: int = 1,
        command_conditioned_grid: bool = False,
    ) -> None:
        self.grid_size = int(grid_size)
        self.grid_history_len = int(grid_history_len)
        self.grid_cell_size = float(grid_cell_size)
        self.grid_dim = self.grid_size * self.grid_size
        self.grid_stack_dim = self.grid_history_len * self.grid_dim
        self.cnn_feature_dim = int(cnn_feature_dim)
        self.gru_hidden_dim = int(gru_hidden_dim)
        self.gru_layers = int(gru_layers)
        self.command_conditioned_grid = bool(command_conditioned_grid)
        self.grid_input_channels = 4 if self.command_conditioned_grid else 1
        activation = _normalize_activation_name(activation)
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
        if self.grid_history_len <= 0:
            raise ValueError("grid_history_len must be positive")
        if self.gru_layers <= 0:
            raise ValueError("gru_layers must be positive")
        if self.obs_dim <= self.grid_stack_dim:
            raise ValueError(
                "OmniCarGridCNNGRUModel expects stacked grid plus low-state inputs; "
                f"got obs_dim={self.obs_dim}, grid_stack_dim={self.grid_stack_dim}"
            )
        axis = (
            torch.arange(self.grid_size, dtype=torch.float32)
            + 0.5
            - float(self.grid_size) / 2.0
        ) * self.grid_cell_size
        grid_x, grid_y = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("_grid_x", grid_x.view(1, 1, self.grid_size, self.grid_size))
        self.register_buffer("_grid_y", grid_y.view(1, 1, self.grid_size, self.grid_size))
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(self.grid_input_channels, 8, kernel_size=5, stride=2, padding=2),
            _activation(activation),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            _activation(activation),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            _activation(activation),
            nn.Flatten(),
            nn.Linear(32 * 10 * 10, self.cnn_feature_dim),
            _activation(activation),
        )
        self.grid_gru = nn.GRU(
            input_size=self.cnn_feature_dim,
            hidden_size=self.gru_hidden_dim,
            num_layers=self.gru_layers,
            batch_first=True,
        )

    def _condition_grid(self, grid: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        if not self.command_conditioned_grid:
            return grid
        planar_command = command[:, :2]
        command_norm = torch.linalg.vector_norm(planar_command, dim=-1, keepdim=True)
        command_dir = planar_command / torch.clamp(command_norm, min=1e-6)
        command_dir = torch.where(command_norm > 1e-6, command_dir, torch.zeros_like(command_dir))
        dir_x = command_dir[:, 0].view(-1, 1, 1, 1)
        dir_y = command_dir[:, 1].view(-1, 1, 1, 1)
        longitudinal = dir_x * self._grid_x + dir_y * self._grid_y
        lateral_abs = torch.abs(-dir_y * self._grid_x + dir_x * self._grid_y)
        command_scale = torch.clamp(command_norm / 2.0, min=0.0, max=1.0).view(-1, 1, 1, 1)
        condition = torch.cat(
            [
                longitudinal.expand(-1, -1, self.grid_size, self.grid_size),
                lateral_abs.expand(-1, -1, self.grid_size, self.grid_size),
                command_scale.expand(-1, -1, self.grid_size, self.grid_size),
            ],
            dim=1,
        )
        condition = condition[:, None].expand(-1, self.grid_history_len, -1, -1, -1)
        return torch.cat([grid, condition], dim=2)

    def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
        del masks, hidden_state
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        raw_flat = torch.cat(obs_list, dim=-1)
        flat = self.obs_normalizer(raw_flat)
        grid = flat[..., : self.grid_stack_dim].reshape(
            -1,
            self.grid_history_len,
            1,
            self.grid_size,
            self.grid_size,
        )
        low_state = flat[..., self.grid_stack_dim :].reshape(grid.shape[0], -1)
        command = raw_flat[..., self.grid_stack_dim : self.grid_stack_dim + 3].reshape(
            grid.shape[0], 3
        )
        conditioned_grid = self._condition_grid(grid, command)
        frame_features = self.grid_encoder(
            conditioned_grid.reshape(
                -1, self.grid_input_channels, self.grid_size, self.grid_size
            )
        ).reshape(grid.shape[0], self.grid_history_len, self.cnn_feature_dim)
        _, hidden = self.grid_gru(frame_features)
        grid_features = hidden[-1]
        return torch.cat([grid_features, low_state], dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            obs_list = [obs[obs_group] for obs_group in self.obs_groups]
            self.obs_normalizer.update(torch.cat(obs_list, dim=-1))  # type: ignore[attr-defined]

    def _get_latent_dim(self) -> int:
        low_state_dim = self.obs_dim - self.grid_stack_dim
        return self.gru_hidden_dim + low_state_dim
