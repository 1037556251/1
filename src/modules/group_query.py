"""Grouping and query pooling for role/feature-point representations."""

from typing import Optional
from pathlib import Path

import torch
from torch import nn


class GroupQuery(nn.Module):
    """Project roles, group consecutive roles, and pool one query per group."""

    def __init__(self, config_path: str = "config/toy_config.yaml"):
        super().__init__()
        import yaml
        config_file = Path(config_path)
        if not config_file.is_absolute() and not config_file.exists():
            config_file = Path(__file__).resolve().parents[2] / "configs" / "toy_config.yaml"
        with config_file.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        model = config.get("model", {})
        self.d_model = int(model.get("d_model", 128))
        self.d_x = int(model.get("d_x", 64))
        self.M = int(model.get("M", 128))
        self.group_size = int(model.get("group_size", 16))
        if self.M != 128:
            raise ValueError(f"GroupQuery requires M=128, got M={self.M}")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        self.input_projection = nn.Linear(self.d_x, self.d_model)

    @property
    def num_groups(self) -> int:
        return (self.M + self.group_size - 1) // self.group_size

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 3 or x.shape[1:] != (self.M, self.d_x):
            raise ValueError(f"expected [batch, {self.M}, {self.d_x}], got {tuple(x.shape)}")

    def input_to_group(self, input_tensor: torch.Tensor,
                       mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return [batch, M, 128] role features enriched by group means."""
        self._validate_input(input_tensor)
        if mask is not None:
            if mask.shape != input_tensor.shape[:2]:
                raise ValueError("mask must have shape [batch, M]")
            valid = mask.to(input_tensor).unsqueeze(-1)
        else:
            valid = torch.ones_like(input_tensor[..., :1])

        projected = self.input_projection(input_tensor) * valid
        padded = self.num_groups * self.group_size - self.M
        if padded:
            projected = torch.cat([projected, projected.new_zeros(projected.shape[0], padded, self.d_model)], 1)
            valid = torch.cat([valid, valid.new_zeros(valid.shape[0], padded, 1)], 1)
        grouped = projected.view(-1, self.num_groups, self.group_size, self.d_model)
        group_valid = valid.view(-1, self.num_groups, self.group_size, 1)
        queries = (grouped * group_valid).sum(2) / group_valid.sum(2).clamp_min(1.0)
        queries_per_role = queries.repeat_interleave(self.group_size, 1)[:, :self.M]
        # Keep role-specific information while making the actual group mean
        # affect every role in that group.
        return (projected[:, :self.M] + queries_per_role) * valid[:, :self.M]

    def group_to_query(self, group_representation: torch.Tensor,
                       query_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Mean-pool each 16-role group, returning [batch, 8, 128]."""
        if group_representation.ndim != 3 or group_representation.shape[-1] != self.d_model:
            raise ValueError("group_representation must be [batch, M, 128]")
        if group_representation.shape[1] == self.M:
            padded = self.num_groups * self.group_size - self.M
            if padded:
                group_representation = torch.cat(
                    [group_representation, group_representation.new_zeros(
                        group_representation.shape[0], padded, self.d_model
                    )], dim=1
                )
            queries = group_representation.view(
                -1, self.num_groups, self.group_size, self.d_model
            ).mean(2)
        elif group_representation.shape[1] == self.num_groups:
            queries = group_representation
        else:
            raise ValueError(f"expected {self.M} roles or {self.num_groups} groups")
        if query_indices is None:
            return queries
        indices = query_indices.to(device=queries.device, dtype=torch.long)
        if indices.ndim == 1:
            return queries[:, indices, :]
        if indices.ndim != 2 or indices.shape[0] != queries.shape[0]:
            raise ValueError("query_indices must be [n_queries] or [batch, n_queries]")
        return torch.gather(queries, 1, indices.unsqueeze(-1).expand(-1, -1, self.d_model))

    def forward(self, input_tensor: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.input_to_group(input_tensor, mask)

    def apply_permutation(self, group_representation: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
        return group_representation[:, permutation.to(group_representation.device), :]
