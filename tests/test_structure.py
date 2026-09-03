"""完整固定资源通信链路的结构测试。"""

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
    """将每个包含 16 个 codec 向量的分组展平，并取其前 q_k 个比特。"""
    role_bits = (codec.reshape(codec.shape[0], 8, 16 * 32) > 0).float()
    return torch.cat([role_bits[:, k, :q] for k, q in enumerate(q_list)], dim=1)


def test_complete_forward_pipeline():
    """测试 Input → Group/Query → Quantizer → profile → 编码 → 信道 → 接收。"""
    torch.manual_seed(7)
    input_mask = InputMask(CONFIG)
    inputs = input_mask.generate_random_tensor(batch_size=2)
    assert inputs.shape == (2, 128, 64)
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
    """验证 13 个 profile 均由头部和 8064 个载荷比特组成。"""
    mother_code = MotherCode(CONFIG)
    profiles = ProfileGenerator(CONFIG).get_all_profiles()
    for profile in profiles:
        source = torch.zeros(1, sum(profile["q"]))
        encoded = mother_code.encode_from_profile(source, profile)
        assert encoded.shape == (1, 8192)
        assert encoded.shape[1] == 128 + sum(profile["n"]) == 128 + 8064


def test_all_profiles_share_one_fixed_parity_graph():
    """验证所有 profile 使用同一张固定尺寸、固定内容的校验图。"""
    mother_code = MotherCode(CONFIG)
    decoder = Receiver(CONFIG).decoder
    assert mother_code.profile_H.shape == (1008, 1264)
    assert decoder.profile_H.shape == (1008, 1264)
    torch.testing.assert_close(
        mother_code.profile_H, decoder.profile_H, rtol=0, atol=0)
    for profile in ProfileGenerator(CONFIG).get_all_profiles():
        assert mother_code._parity_projection(
            max(profile["q"]), max(n - q for q, n in zip(profile["q"], profile["n"]))
        ).shape == (256, 1008)


def test_noiseless_round_trip_is_exact():
    """验证无穷 SNR 传输能够保留 profile 的全部信息比特。"""
    torch.manual_seed(8)
    profile = ProfileGenerator(CONFIG).get_profile(9)
    mother_code = MotherCode(CONFIG)
    channel = SoftwareChannel(CONFIG)
    source = torch.randint(0, 2, (3, sum(profile["q"]))).float()
    encoded = mother_code.encode_from_profile(source, profile)
    received = channel.transmit(encoded, snr_db=float("inf"))
    decoded = Receiver(CONFIG).decoder.decode(received, 8192, sum(profile["q"]))
    assert torch.max(torch.abs(decoded - source)).item() < 1e-6


def test_profile_decoder_corrects_a_single_bit_error():
    """验证 profile 校验图能够纠正一个发送比特错误。"""
    torch.manual_seed(10)
    profile = ProfileGenerator(CONFIG).get_profile(0)
    mother_code = MotherCode(CONFIG)
    source = torch.randint(0, 2, (1, sum(profile["q"]))).float()
    encoded = mother_code.encode_from_profile(source, profile)
    corrupted = encoded.clone()
    corrupted[:, 128 + 100] = 1.0 - corrupted[:, 128 + 100]
    decoded = Receiver(CONFIG).decoder.decode(
        corrupted, 8192, sum(profile["q"])
    )
    torch.testing.assert_close(decoded, source, rtol=0, atol=0)


def test_fixed_seed_noise_is_reproducible():
    """验证使用配置中的信道种子会产生完全相同的含噪观测。"""
    bits = torch.randint(0, 2, (2, 1024)).float()
    first = SoftwareChannel(CONFIG).transmit(bits)
    second = SoftwareChannel(CONFIG).transmit(bits)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_synchronized_group_profile_and_embedding_permutation():
    """验证整组置换并执行逆置换后，所有等变输出保持一致。"""
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
