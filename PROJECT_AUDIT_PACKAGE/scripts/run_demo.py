"""Run a random-tensor demo through the complete communication pipeline."""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from modules.group_query import GroupQuery
from modules.mother_code import MotherCode
from modules.profile_generator import ProfileGenerator
from modules.quantizer import Quantizer
from modules.receiver import Receiver
from modules.software_channel import SoftwareChannel


def main() -> None:
    config = ROOT / "configs" / "toy_config.yaml"
    torch.manual_seed(123)
    x = torch.randn(2, 128, 64)
    print(f"Input:               {tuple(x.shape)}")

    group_features = GroupQuery(str(config))(x)
    print(f"Group/Query features: {tuple(group_features.shape)}")
    codec = Quantizer(str(config))(group_features)
    print(f"Quantizer codec:      {tuple(codec.shape)}")

    profile = ProfileGenerator(str(config)).get_profile(0)
    role_bits = (codec.reshape(2, 8, 16 * 32) > 0).float()
    source = torch.cat([role_bits[:, k, :q] for k, q in enumerate(profile["q"])], dim=1)
    print(f"Profile 0:            q={profile['q']}, n={profile['n']}")
    encoded = MotherCode(str(config)).encode_from_profile(source, profile)
    print(f"Mother-code output:   {tuple(encoded.shape)}")

    channel = SoftwareChannel(str(config))
    received, soft = channel.transmit(encoded, return_soft=True)
    print(f"Channel hard bits:     {tuple(received.shape)}")
    print(f"Channel soft values:   {tuple(soft.shape)}, SNR={channel.snr_db} dB")
    decoded, stats = Receiver(str(config)).receive(
        received, 8192, sum(profile["q"]), soft_information=soft
    )
    print(f"Receiver decoded:      {tuple(decoded.shape)}")
    print(f"Decoding failed:       {stats['decoding_failed']}")


if __name__ == "__main__":
    main()
