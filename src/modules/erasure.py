"""Unified erasure detection based on received soft information."""

import os
from typing import Optional
import torch


def erasure_function(received_bits: torch.Tensor,
                     soft_information: Optional[torch.Tensor] = None,
                     threshold: float = 0.5) -> torch.Tensor:
    """Mark unreliable received bits; transmitted/source bits are never used."""
    if received_bits.ndim != 2:
        raise ValueError("received_bits must be [batch, n_bits]")
    if soft_information is None:
        return torch.zeros_like(received_bits, dtype=torch.bool)
    if soft_information.shape != received_bits.shape:
        raise ValueError("soft_information must match received_bits shape")
    return soft_information.abs() < threshold


class Erasure:
    def __init__(self, config_path: str = None):
        import yaml
        if config_path is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(module_dir, '..', '..', 'configs', 'toy_config.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        cfg = self.config.get('erasure', {})
        self.erasure_probability = cfg.get('probability', 0.1)
        self.soft_threshold = cfg.get('soft_threshold', 0.5)
        self.erasure_pattern = None

    def erasure_function(self, received_bits: torch.Tensor,
                         soft_information: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.erasure_pattern = erasure_function(received_bits, soft_information,
                                                 self.soft_threshold)
        return self.erasure_pattern

    def apply_erasure(self, bits: torch.Tensor, probability: Optional[float] = None) -> torch.Tensor:
        """Legacy random helper retained for compatibility; not used by receive()."""
        if probability is None:
            probability = self.erasure_probability
        mask = torch.rand(bits.shape, device=bits.device) < probability
        self.erasure_pattern = mask
        erased = bits.clone()
        erased[mask] = -1
        return erased

    def apply_pattern_erasure(self, bits: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
        pattern = pattern.to(device=bits.device, dtype=torch.bool)
        if pattern.ndim == 1:
            pattern = pattern.unsqueeze(0).expand(bits.shape[0], -1)
        if pattern.shape != bits.shape:
            raise ValueError("pattern must match bits shape")
        self.erasure_pattern = pattern
        erased = bits.clone()
        erased[pattern] = -1
        return erased

    def recover_erasure(self, erased_bits: torch.Tensor,
                        known_bits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Keep marks in the receiver path; retain old explicit recovery API.

        ``known_bits`` is accepted only for backward compatibility with the
        old standalone helper tests.  The real receive path never supplies it,
        so noisy received bits are never replaced by source/truth bits.
        """
        if known_bits is not None:
            recovered = erased_bits.clone()
            recovered[erased_bits == -1] = known_bits[erased_bits == -1]
            return recovered
        return erased_bits

    def get_erasure_info(self) -> dict:
        return {'erasure_probability': self.erasure_probability,
                'soft_threshold': self.soft_threshold,
                'erasure_pattern': self.erasure_pattern}
