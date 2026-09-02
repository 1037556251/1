"""将角色特征可学习地投影为 codec 向量。"""

import torch
from torch import nn
from pathlib import Path


class Quantizer(nn.Module):
    """将 d_x 维输入投影到 128 维，再量化到 codec_dim；已是 128 维时跳过首层投影。"""

    def __init__(self, config_path: str = "config/toy_config.yaml"):
        super().__init__()
        import yaml
        config_file = Path(config_path)
        if not config_file.is_absolute() and not config_file.exists():
            config_file = Path(__file__).resolve().parents[2] / "configs" / "toy_config.yaml"
        with config_file.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        model = config.get("model", {})
        self.d_x = int(model.get("d_x", 64))
        self.d_model = int(model.get("d_model", 128))
        self.codec_dim = int(model.get("codec_dim", 32))
        self.codebook_size = int(model.get("codebook_size", 256))
        self.input_projection = nn.Linear(self.d_x, self.d_model)
        self.codec_projection = nn.Linear(self.d_model, self.codec_dim)
        self.reconstruction_projection = nn.Linear(self.codec_dim, self.d_model)
        self.codebook = nn.Parameter(torch.randn(self.codebook_size, self.codec_dim))

    def quantize(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """从 [batch, M, d_x] 或 [batch, M, 128] 返回 [batch, M, 32]。"""
        if input_tensor.ndim != 3:
            raise ValueError("input_tensor must be a 3-D tensor")
        if input_tensor.shape[-1] == self.d_x:
            hidden = self.input_projection(input_tensor)
        elif input_tensor.shape[-1] == self.d_model:
            hidden = input_tensor
        else:
            raise ValueError(f"last dimension must be {self.d_x} or {self.d_model}")
        return self.codec_projection(hidden)

    forward = quantize

    def dequantize(self, codec_vector: torch.Tensor) -> torch.Tensor:
        if codec_vector.ndim != 3 or codec_vector.shape[-1] != self.codec_dim:
            raise ValueError(f"codec_vector must be [batch, M, {self.codec_dim}]")
        return self.reconstruction_projection(codec_vector)

    def get_codebook(self) -> torch.Tensor:
        return self.codebook
