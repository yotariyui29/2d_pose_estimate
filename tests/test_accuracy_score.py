"""タスク精度スコア(CE・VE・TE・測定誤差の感度・要件)の検証: python tests/test_accuracy_score.py (WSL venv)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pose2d.accuracy_score import D, deg, noise_effect, requirements, scores, tracking_requirement, ve_power


def test_score_identity() -> None:
    rng = np.random.default_rng(0)
    ey, ez = rng.normal(0.4, 1.0, 12), rng.normal(-0.2, 0.5, 12)
    s = scores(ey, ez)
    assert abs(s["TE"] ** 2 - (s["CE"] ** 2 + s["VE"] ** 2 * (len(ey) - 1) / len(ey))) < 1e-9  # TE² = CE² + VE²(N-1)/N
    assert abs(s["VE"] - np.hypot(ey.std(ddof=1), ez.std(ddof=1))) < 1e-12
    assert abs(deg(D * np.tan(np.radians(3.0))) - 3.0) < 1e-9


def test_noise_effect() -> None:
    assert abs(noise_effect(1.0, 0.1) - np.sqrt(1.02)) < 1e-12 and noise_effect(0.5, 0.02) < 1.002


def test_requirements() -> None:
    sens = pd.DataFrame(dict(input=["az", "elev", "h_lat", "h_up"], dy_k_m_per_step=[0.192, 0.0, 0.5916, -0.0051], dz_m_per_step=[0.0, 0.192, -0.0077, 0.5635]))
    r = requirements(sens, budgets=(0.2,)).set_index("input")["required_sd"]
    assert abs(r["foot velocity azimuth (deg)"] - 0.2 / (2 * 0.192)) < 1e-9
    assert abs(r["contact offset, lateral (mm)"] - 0.2 / (2 * 59.16) * 1000) < 1e-6
    t = tracking_requirement(0.5, 18.0, (5,))
    assert abs(t["required_position_sd_mm"].iloc[0] - np.tan(np.radians(0.5)) * 90 / np.sqrt(2)) < 1e-9


def test_power_monotone() -> None:
    p = ve_power(ns=(10, 30, 50), ratios=(1.5,))
    for _, g in p.groupby("shape"):
        assert (np.diff(g.sort_values("n_per_subject")["power"]) > 0).all()
    assert (p[p["shape"].str.startswith("iso")].power.to_numpy() > p[p["shape"].str.startswith("elong")].power.to_numpy()).all()


if __name__ == "__main__":
    test_score_identity()
    test_noise_effect()
    test_requirements()
    test_power_monotone()
    print("OK")
