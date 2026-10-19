"""試行間依存(ACF(1)・補正係数行列K・シャッフル検定)の検証: python tests/test_sequential.py (WSL venv)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pose2d.sequential import acf1, analyze, lag_matrix


def _ar(n: int, A: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    e = np.zeros((n, 2))
    for i in range(1, n):
        e[i] = A @ e[i - 1] + rng.normal(0, 1, 2)
    return e[:, 0], e[:, 1]


def test_lag_matrix_recovers_A() -> None:
    A = np.array([[-0.5, 0.2], [-0.3, 0.1]])
    y, z = _ar(20000, A, 0)
    assert np.abs(lag_matrix(y, z) - A).max() < 0.04


def test_acf_sign_and_null() -> None:
    y, z = _ar(20000, np.array([[-0.5, 0.0], [0.0, 0.5]]), 1)
    assert acf1(y) < -0.4 and acf1(z) > 0.4
    rng = np.random.default_rng(2)
    x = rng.normal(size=8)  # 短い独立系列: ACF(1) の期待値は 0 より小さい側へ偏る(約 -1/(n-1))
    null = [acf1(rng.permutation(x)) for _ in range(4000)]
    assert -0.25 < np.mean(null) < -0.05, np.mean(null)


def test_analyze_detects_dependence() -> None:
    rng = np.random.default_rng(3)
    series = []
    for k in range(3):
        y, z = _ar(10, np.array([[-0.8, 0.0], [0.0, 0.0]]), 10 + k)
        series.append(dict(session=f"s{k}", sid=f"ID{k}", exp=10.0, no=np.arange(1, 11), y=y, z=z + rng.normal(0, 0.0, 10)))
    acf, k = analyze(series, n_perm=2000, seed=0)
    a = acf[(acf["group"] == "ALL") & (acf["axis"] == "y")].iloc[0]
    assert a["acf1"] < a["null_lo"] and a["p_neg"] < 0.05, a
    assert (acf["group"] == "EXP").any()
    assert len(k[k["group"] == "ALL"]) == 4


def test_detrend_null_is_calibrated() -> None:
    """傾向のある独立な系列を傾向除去で解析したとき、「負側 p<0.05」になる割合が約5%(残差だけをシャッフルすると約10%以上になる)。"""
    rng = np.random.default_rng(5)
    hit = 0
    reps = 150
    for _ in range(reps):
        n = 10
        s = dict(session="s", sid="ID", exp=1.0, no=np.arange(1, n + 1), y=0.2 * np.arange(n) + rng.normal(size=n), z=rng.normal(size=n))
        acf, _k = analyze([s], n_perm=300, seed=int(rng.integers(1e9)), detrend_on=True)
        hit += float(acf[(acf["group"] == "s") & (acf["axis"] == "y")].iloc[0]["p_neg"]) < 0.05
    assert hit / reps < 0.10, hit / reps


if __name__ == "__main__":
    test_detrend_null_is_calibrated()
    test_lag_matrix_recovers_A()
    test_acf_sign_and_null()
    test_analyze_detects_dependence()
    print("OK")
