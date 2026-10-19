"""前の誤差→つまみ→次の誤差の媒介解析の検証: python tests/test_mediation.py (WSL venv)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pose2d.mediation import analyze, path_stats, rank_normal


def _series(n: int, seed: int, link: bool) -> dict:
    """k(n+1) = -0.8 e_y(n) + 雑音、e_y(n+1) = 0.8 k(n+1) + 雑音(link=False なら e は k と無関係)。"""
    rng = np.random.default_rng(seed)
    e = np.zeros(n)
    k = np.zeros(n)
    for i in range(1, n):
        k[i] = (-0.8 * e[i - 1] if link else 0.0) + rng.normal(0, 0.6)
        e[i] = (0.8 * k[i] if link else 0.0) + rng.normal(0, 0.6)
    z = rng.normal(size=n)
    return dict(session="s", sid="ID", exp=10.0, no=np.arange(1, n + 1), E=np.stack([rank_normal(e), rank_normal(z)], -1),
                knobs={"toe_vel_elev_deg": rank_normal(k)})


def test_path_recovers_chain() -> None:
    s = _series(4000, 0, True)
    st = path_stats(s["E"][None], s["knobs"]["toe_vel_elev_deg"][None])
    assert st["a"][0, 0] < -0.6 and st["b"][0, 0] > 0.4, st
    assert st["ind"][0, 0, 0] < -0.25


def test_regression_to_mean_is_not_correction() -> None:
    """e(n) の原因が k(n) で k が独立なら、k(n+1) は e(n) に反応しない(a≈0)。"""
    rng = np.random.default_rng(1)
    n = 6000
    k = rng.normal(size=n)
    e = k + rng.normal(0, 0.5, n)
    E = np.stack([rank_normal(e), rank_normal(rng.normal(size=n))], -1)
    st = path_stats(E[None], rank_normal(k)[None])
    assert abs(st["a"][0, 0]) < 0.06, st["a"]


def test_analyze_detects_and_null_is_quiet() -> None:
    df = analyze([_series(11, 10 + i, True) for i in range(4)], n_perm=1500, seed=0)
    r = df[(df.group == "ALL") & (df.err == "y") & (df.next_err == "y")].iloc[0]
    assert r["ind"] < -0.1 and r["p_ind"] < 0.1, r
    df0 = analyze([_series(11, 50 + i, False) for i in range(4)], n_perm=1500, seed=0)
    r0 = df0[(df0.group == "ALL") & (df0.err == "y") & (df0.next_err == "y")].iloc[0]
    assert r0["p_ind"] > 0.05, r0


if __name__ == "__main__":
    test_path_recovers_chain()
    test_regression_to_mean_is_not_correction()
    test_analyze_detects_and_null_is_quiet()
    print("OK")
