"""要因表の補助関数の検証: python tests/test_factors.py (WSL venv)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pose2d.kickfactors import tilt_deg


def test_tilt() -> None:
    rng = np.random.default_rng(0)
    s = rng.normal(0, 1.0, 400)
    n = rng.normal(0, 0.1, 400)
    a = np.radians(30)
    y, z = s * np.cos(a) - n * np.sin(a), s * np.sin(a) + n * np.cos(a)
    assert abs(tilt_deg(y, z) - 30) < 3, tilt_deg(y, z)
    assert abs(tilt_deg(y, -z) + 30) < 3  # 反転すると負の傾き


if __name__ == "__main__":
    test_tilt()
    print("OK")
