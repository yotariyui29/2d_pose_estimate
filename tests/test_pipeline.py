"""合成データによる検証: python tests/test_pipeline.py (WSL venv)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pose2d.calib import Calibration
from pose2d.config import ProcessCfg
from pose2d.keypoints import OUTPUT_KPS
from pose2d.process import process, winter_cutoff

FPS, N, IMPACT = 1000.0, 600, 400


def synth(cal: Calibration, noise_px=1.0, seed=0) -> tuple[pd.DataFrame, np.ndarray]:
    """右脚の振り子キック。真値(m)を返す。"""
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FPS
    th = -1.2 + 2.2 * (1 - np.exp(-((t / 0.4) ** 2))) * 0 + 2.2 * (t / (IMPACT / FPS)) ** 2  # 大腿角(rad)
    hip = np.array([1.0, 0.9])
    knee = hip + 0.45 * np.column_stack([np.sin(th), -np.cos(th)])
    ank = knee + 0.45 * np.column_stack([np.sin(th * 1.4), -np.cos(th * 1.4)])
    toe = ank + 0.2 * np.column_stack([np.cos(th * 1.4), np.sin(th * 1.4)])
    truth = {"right_hip": np.tile(hip, (N, 1)), "right_knee": knee, "right_ankle": ank, "right_big_toe": toe}
    ball = np.tile([1.6, 0.11], (N, 1)).astype(float)
    ball[IMPACT + 1:, 0] += 25.0 * (np.arange(N - IMPACT - 1) + 1) / FPS
    truth["ball"] = ball

    # m → px は Calibration の逆変換
    M = np.linalg.inv(cal._inv)
    def to_px(p):
        p = p.copy(); p[:, 0] *= (-1 if cal.mirror_x else 1)
        return p @ M.T + cal._o
    raw = {"frame": np.arange(N)}
    for k in OUTPUT_KPS:
        if k in truth:
            px = to_px(truth[k]) + rng.normal(0, noise_px, (N, 2))
            raw[f"{k}_x"], raw[f"{k}_y"], raw[f"{k}_c"] = px[:, 0], px[:, 1], np.full(N, 0.9)
        else:
            raw[f"{k}_x"] = raw[f"{k}_y"] = raw[f"{k}_c"] = np.full(N, np.nan)
    bpx = to_px(ball)
    raw["ball_x"], raw["ball_y"], raw["ball_c"] = bpx[:, 0], bpx[:, 1], np.full(N, 0.9)
    df = pd.DataFrame(raw)
    df.loc[200:210, "right_knee_c"] = 0.05       # 欠損
    df.loc[300, "right_ankle_x"] += 200          # スパイク
    return df, truth


def main() -> None:
    # 傾き・縦横比の異なる定規: O=(300,900), X端=(300+1.1*380, 905), Y端=(302, 900-1.0*380)
    cal = Calibration((300, 900), (300 + 1.1 * 380, 905), (302, 900 - 380))
    assert abs(cal.mm_per_px_x - 1000 / 380) < 0.05, cal.mm_per_px_x
    q = cal.px_to_m(np.array([[300 + 1.1 * 380, 905.0], [302, 520.0], [300, 900.0]]))
    assert np.allclose(q, [[1.1, 0], [0, 1.0], [0, 0]], atol=1e-6), q
    print("calib OK:", cal.summary())

    raw, truth = synth(cal)
    kin, info = process(raw, cal, FPS, ProcessCfg(cutoff_hz=20.0), 0.3, "right", None)
    err = np.nanmean(np.linalg.norm(kin[["right_ankle_x", "right_ankle_y"]].to_numpy() - truth["right_ankle"], axis=1)[50:-50])
    print(f"ankle mean err {err*1000:.2f} mm | info: {info}")
    assert err < 0.01, err
    assert abs(info["impact_frame"] - IMPACT) <= 3, info["impact_frame"]
    assert abs(info["ball_speed_mps"] - 25.0) < 1.0, info["ball_speed_mps"]
    assert kin["right_knee_x"].notna().all(), "短い欠損が補間されていない"
    assert abs(kin.loc[300, "right_ankle_x"] - truth["right_ankle"][300, 0]) < 0.02, "スパイク未除去"

    # 手動インパクト・auto遮断
    _, info2 = process(raw, cal, FPS, ProcessCfg(cutoff_hz="auto"), 0.3, "right", IMPACT)
    print("auto cutoff:", round(info2["cutoff_hz"], 1), "Hz")
    assert 3 < info2["cutoff_hz"] < 100 and info2["impact_frame"] == IMPACT
    print("ALL OK")


if __name__ == "__main__":
    main()
