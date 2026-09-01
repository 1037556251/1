"""Structural tests for the complete fixed-resource communication path."""

import numpy as np
import torch
from pathlib import Path

from modules.group_query import GroupQuery
from modules.input_mask import InputMask
from modules.mother_code import MotherCode
from modules.profile_generator import ProfileGenerator
from modules.quantizer import Quantizer
from modules.receiver import Receiver
from modules.software_channel import SoftwareChannel


CONFIG = str(Path(__file__).resolve().parents[1] / "configs" / "toy_config.yaml")


def _codec_source(codec: torch.Tensor, q_list) -> torch.Tensor:
    """Flatten each block of 16 codec vectors and take its q_k leading bits."""
    role_bits = (codec.reshape(codec.shape[0], 8, 16 * 32) > 0).float()
    return torch.cat([role_bits[:, k, :q] for k, q in enumerate(q_list)], dim=1)


def test_complete_forward_pipeline():
    """Input → Group/Query → Quantizer → profile → code → channel → receive."""
    torch.manual_seed(7)
    inputs = torch.randn(2, 128, 64)
    input_mask = InputMask(CONFIG)
    profile = ProfileGenerator(CONFIG).get_profile(0)
    group_query = GroupQuery(CONFIG)
    quantizer = Quantizer(CONFIG)
    mother_code = MotherCode(CONFIG)
    channel = SoftwareChannel(CONFIG)
    receiver = Receiver(CONFIG)

    group_features = group_query(inputs, input_mask.generate_source_mask(profile, 2)[:, :, :128].any(1))
    assert group_features.shape == (2, 128, 128)
    codec = quantizer(group_features)
    assert codec.shape == (2, 128, 32)
    source = _codec_source(codec, profile["q"])
    encoded = mother_code.encode_from_profile(source, profile)
    received, soft = channel.transmit(encoded, return_soft=True)
    decoded, stats = receiver.receive(received, 8192, sum(profile["q"]), soft_information=soft)

    assert encoded.shape == received.shape == (2, 8192)
    assert decoded.shape == (2, sum(profile["q"]))
    assert stats["final_shape"] == decoded.shape


def test_all_profiles_have_8192_output_bits():
    """Every one of the 13 profiles occupies header plus 8064 payload bits."""
    mother_code = MotherCode(CONFIG)
    profiles = ProfileGenerator(CONFIG).get_all_profiles()
    for profile in profiles:
        source = torch.zeros(1, sum(profile["q"]))
        encoded = mother_code.encode_from_profile(source, profile)
        assert encoded.shape == (1, 8192)
        assert encoded.shape[1] == 128 + sum(profile["n"]) == 128 + 8064


def test_noiseless_round_trip_is_exact():
    """Infinite-SNR transmission preserves all profile information bits."""
    torch.manual_seed(8)
    profile = ProfileGenerator(CONFIG).get_profile(9)
    mother_code = MotherCode(CONFIG)
    channel = SoftwareChannel(CONFIG)
    source = torch.randint(0, 2, (3, sum(profile["q"]))).float()
    encoded = mother_code.encode_from_profile(source, profile)
    received = channel.transmit(encoded, snr_db=float("inf"))
    payload = received[:, 128:]
    decoded = torch.cat([
        payload[:, sum(profile["n"][:k]):sum(profile["n"][:k + 1])][:, :profile["q"][k]]
        for k in range(8)
    ], dim=1)
    assert torch.max(torch.abs(decoded - source)).item() < 1e-6


def test_fixed_seed_noise_is_reproducible():
    """The configured channel seed produces identical noisy observations."""
    bits = torch.randint(0, 2, (2, 1024)).float()
    first = SoftwareChannel(CONFIG).transmit(bits)
    second = SoftwareChannel(CONFIG).transmit(bits)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_synchronized_group_profile_and_embedding_permutation():
    """Permuting whole groups and reversing it preserves all equivariant outputs."""
    torch.manual_seed(9)
    group_query = GroupQuery(CONFIG)
    quantizer = Quantizer(CONFIG)
    x = torch.randn(2, 128, 64)
    permutation = torch.tensor([2, 0, 3, 1, 6, 4, 7, 5])
    inverse = torch.argsort(permutation)

    original_embedding = quantizer(group_query(x))
    permuted_x = x.reshape(2, 8, 16, 64)[:, permutation].reshape(2, 128, 64)
    permuted_embedding = quantizer(group_query(permuted_x))
    restored_embedding = permuted_embedding.reshape(2, 8, 16, 32)[:, inverse].reshape(2, 128, 32)
    torch.testing.assert_close(original_embedding, restored_embedding, rtol=1e-6, atol=1e-6)

    profile = ProfileGenerator(CONFIG).get_profile(0)
    mask = torch.tensor(profile["mask"], dtype=torch.float32).reshape(8, 16)
    restored_mask = mask[permutation][inverse]
    np.testing.assert_array_equal(mask.numpy(), restored_mask.numpy())
