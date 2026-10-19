"""確認用の重畳動画と時系列プロット。"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .detect import iter_frames_idx
from .figio import save_figure

# 部位ごとの色 (BGR)
PART_COLORS = {
    "head": (255, 255, 255),
    "trunk": (0, 230, 255),      # 黄
    "arm_L": (255, 220, 0),      # 水色
    "arm_R": (255, 0, 220),      # マゼンタ
    "leg_L": (0, 220, 0),        # 緑
    "leg_R": (255, 90, 0),       # 青
    "ball": (0, 0, 255),         # 赤
}
PART_LABELS = {"head": "Head", "trunk": "Trunk", "arm_L": "L arm", "arm_R": "R arm",
               "leg_L": "L leg", "leg_R": "R leg", "ball": "Ball"}

# (点A, 点B, 部位)
_BONES = [
    ("head", "neck", "head"), ("nose", "head", "head"),
    ("neck", "left_shoulder", "trunk"), ("neck", "right_shoulder", "trunk"),
    ("neck", "pelvis", "trunk"), ("pelvis", "left_hip", "trunk"), ("pelvis", "right_hip", "trunk"),
    ("left_shoulder", "left_elbow", "arm_L"), ("left_elbow", "left_wrist", "arm_L"),
    ("right_shoulder", "right_elbow", "arm_R"), ("right_elbow", "right_wrist", "arm_R"),
]
for _s, _p in (("left", "leg_L"), ("right", "leg_R")):
    _BONES += [(f"{_s}_hip", f"{_s}_knee", _p), (f"{_s}_knee", f"{_s}_ankle", _p),
               (f"{_s}_ankle", f"{_s}_heel", _p), (f"{_s}_ankle", f"{_s}_big_toe", _p),
               (f"{_s}_heel", f"{_s}_big_toe", _p), (f"{_s}_big_toe", f"{_s}_small_toe", _p)]

# 点の色(その点で終わる骨の部位)。体幹の関節(肩・股・骨盤・首)は体幹色
_POINT_PART = {"nose": "head", "head": "head", "neck": "trunk", "pelvis": "trunk",
               "left_shoulder": "trunk", "right_shoulder": "trunk", "left_hip": "trunk", "right_hip": "trunk"}
for _s, _a, _l in (("left", "arm_L", "leg_L"), ("right", "arm_R", "leg_R")):
    _POINT_PART.update({f"{_s}_elbow": _a, f"{_s}_wrist": _a})
    _POINT_PART.update({f"{_s}_{j}": _l for j in ("knee", "ankle", "heel", "big_toe", "small_toe")})


def _draw_pose(img: np.ndarray, r: pd.Series, min_conf: float = 0.3) -> None:
    """全身の骨格(部位別の色)と関節点を描く。信頼度が低い点は白抜きの輪にする。"""
    def pt(k):
        x, y = r.get(f"{k}_x", np.nan), r.get(f"{k}_y", np.nan)
        return (int(x), int(y)) if np.isfinite([x, y]).all() else None

    for a, b, part in _BONES:
        pa, pb = pt(a), pt(b)
        if pa and pb:
            cv2.line(img, pa, pb, PART_COLORS[part], 5, cv2.LINE_AA)
    for k, part in _POINT_PART.items():
        p = pt(k)
        if p is None:
            continue
        if r.get(f"{k}_c", 1.0) >= min_conf:
            cv2.circle(img, p, 11, (0, 0, 0), -1, cv2.LINE_AA)           # 縁取り
            cv2.circle(img, p, 8, PART_COLORS[part], -1, cv2.LINE_AA)
        else:
            cv2.circle(img, p, 10, PART_COLORS[part], 2, cv2.LINE_AA)    # 低信頼=輪だけ


def _draw_legend(img: np.ndarray) -> None:
    for i, (part, col) in enumerate(PART_COLORS.items()):
        y = 110 + i * 42
        cv2.circle(img, (46, y), 13, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, (46, y), 10, col, -1, cv2.LINE_AA)
        cv2.putText(img, PART_LABELS[part], (72, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, PART_LABELS[part], (72, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)


def overlay_video(video: Path, raw: pd.DataFrame, out: Path, impact: int | None,
                  play_fps: float = 30.0, scale: float = 0.5,
                  frames: tuple[int, int] | None = None, min_conf: float = 0.3) -> None:
    """生検出(px)の全身骨格・関節点・ボールを重畳したスロー再生動画。impactフレームにマーカー。"""
    writer = None
    rows = raw.set_index("frame")
    for f, frame in iter_frames_idx(video, frames):
        if f not in rows.index:
            continue
        r = rows.loc[f]
        _draw_pose(frame, r, min_conf)
        if np.isfinite(r.get("ball_x", np.nan)):
            c = (int(r["ball_x"]), int(r["ball_y"]))
            cv2.circle(frame, c, 16, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.circle(frame, c, 16, PART_COLORS["ball"], 3, cv2.LINE_AA)
        tag = f"frame {f}" + ("  IMPACT" if impact is not None and f == impact else "")
        cv2.putText(frame, tag, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 8, cv2.LINE_AA)
        cv2.putText(frame, tag, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4, cv2.LINE_AA)
        _draw_legend(frame)
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if writer is None:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), play_fps,
                                     (frame.shape[1], frame.shape[0]))
        writer.write(frame)
    if writer:
        writer.release()


def plot_summary(k: pd.DataFrame, info: dict, out: Path, title: str) -> None:
    side = info["kick_side"]
    s = side[0].upper()
    rel = "t_rel_impact_s" in k
    x = k["t_rel_impact_s"] if rel else k["time_s"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for col, lab in ((f"{side}_big_toe_speed", "toe"), (f"{side}_ankle_speed", "ankle"),
                     (f"{side}_knee_speed", "knee"), ("ball_speed", "ball")):
        if col in k:
            ax[0, 0].plot(x, k[col], label=lab)
    ax[0, 0].set_ylabel("speed [m/s]"); ax[0, 0].legend()
    for col, lab in ((f"thigh_{s}_omega_dps", "thigh"), (f"shank_{s}_omega_dps", "shank"),
                     (f"foot_{s}_omega_dps", "foot")):
        if col in k:
            ax[0, 1].plot(x, k[col], label=lab)
    ax[0, 1].set_ylabel("segment angular velocity [deg/s]"); ax[0, 1].legend()
    if f"knee_flexion_{s}_deg" in k:
        ax[1, 0].plot(x, k[f"knee_flexion_{s}_deg"])
    ax[1, 0].set_ylabel("knee flexion [deg]")
    for j in ("hip", "knee", "ankle", "big_toe"):
        if f"{side}_{j}_y" in k:
            ax[1, 1].plot(x, k[f"{side}_{j}_y"], label=j)
    ax[1, 1].set_ylabel("height y [m]"); ax[1, 1].legend()
    for a in ax.ravel():
        a.axvline(0, color="r", ls=":"); a.grid(alpha=.3)
    for a in ax[1]:
        a.set_xlabel("time from impact [s]" if rel else "time [s]")
    bs = info.get("ball_speed_mps")
    fig.suptitle(f"{title}  fc={info['cutoff_hz']:.1f}Hz  ball={'n/a' if bs is None else f'{bs:.1f} m/s'}")
    fig.tight_layout(); save_figure(fig, out, dpi=110); plt.close(fig)
