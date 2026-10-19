"""3D運動学(output/imu3d/<試行>/kinematics3d.csv)の骨格アニメーション(mp4)。  python run.py animate [--trials t1,t2] [--speed 0.2]

左: 側面(カメラ視点, x-z) / 中: 上面(x-y) / 右: 3D(斜め)。蹴り脚は赤、支持脚は青、骨盤は黒、蹴り脚のつま先の軌跡(直前0.2秒)を細線で描く。
インパクト(τ=0)の瞬間に「IMPACT」を表示。再生は実時間の --speed 倍(既定0.2 = 5倍スロー)。英語版 animation.mp4 と日本語版 animation_ja.mp4 を出す。
座標: x=画像右, y=奥行き, z=鉛直上(imu_video_3d の W 系)。3D位置は股関節のみ2Dで固定した推定で、奥行きは仮定を含む(check.png の誤差を参照)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pose2d import i18n

from .imu_video_3d import OUT

SIDES = ("L", "R")
TRAIL_S = 0.2


def _arr(df: pd.DataFrame, key: str) -> np.ndarray:
    return df[[f"{key}_x", f"{key}_y", f"{key}_z"]].to_numpy(float)


def render(df: pd.DataFrame, kick: str, title: str, path: Path, speed: float = 0.2, ja: bool = False, fps: int = 20) -> Path:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = i18n.translate if ja else (lambda s: s)
    font = i18n.japanese_font() if ja else None
    if ja and font is None:
        raise RuntimeError("日本語フォントがありません")
    tau = df["tau"].to_numpy()
    P = {f"{j}{s}": _arr(df, f"{j}{s}") for s in SIDES for j in ("hip", "knee", "ankle", "toe")}
    allp = np.vstack(list(P.values()))
    lo, hi = allp.min(0) - 0.15, allp.max(0) + 0.15
    support = "R" if kick == "L" else "L"
    dt = float(np.median(np.diff(tau)))
    frames = np.arange(0, len(tau), max(1, int(round(speed / fps / dt))))  # 動画1コマ = 実時間 speed/fps 秒 = データ speed/fps/dt サンプル

    with matplotlib.rc_context({"font.family": [font]} if font else {}):
        fig = plt.figure(figsize=(11, 3.8), dpi=80)
        a1, a2 = fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2)
        a3 = fig.add_subplot(1, 3, 3, projection="3d")
        for a, (i, j), ttl in ((a1, (0, 2), "Side view (camera)"), (a2, (0, 1), "Top view")):
            a.set_xlim(lo[i], hi[i]); a.set_ylim(lo[j], hi[j]); a.set_aspect("equal"); a.grid(alpha=0.3)
            a.set_title(T(ttl), fontsize=9)
            a.set_xlabel("x [m]", fontsize=8); a.set_ylabel(("z" if j == 2 else "y") + " [m]", fontsize=8); a.tick_params(labelsize=7)
        a3.set_xlim(lo[0], hi[0]); a3.set_ylim(lo[1], hi[1]); a3.set_zlim(lo[2], hi[2])
        a3.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])); a3.view_init(elev=18, azim=-60)
        a3.set_title(T("3D view"), fontsize=9); a3.tick_params(labelsize=6)
        col = {kick: "#d62d2d", support: "#2a78d6"}
        arts = []  # (ax, 2D/3D, line, joint-keys)
        for s in SIDES:
            keys = [f"hip{s}", f"knee{s}", f"ankle{s}", f"toe{s}"]
            for a in (a1, a2, a3):
                (ln,) = a.plot([], [], [], "-o", color=col[s], lw=2.2, ms=3, label=T("kicking leg" if s == kick else "support leg")) if a is a3 \
                    else a.plot([], [], "-o", color=col[s], lw=2.2, ms=3, label=T("kicking leg" if s == kick else "support leg"))
                arts.append((a, ln, keys))
        for a in (a1, a2, a3):
            (ln,) = a.plot([], [], [], "-", color="k", lw=2.2) if a is a3 else a.plot([], [], "-", color="k", lw=2.2)
            arts.append((a, ln, ["hipL", "hipR"]))
            (tr,) = a.plot([], [], [], ":", color="#d62d2d", lw=1.0) if a is a3 else a.plot([], [], ":", color="#d62d2d", lw=1.0)
            arts.append((a, tr, [f"toe{kick}"]))
        a1.legend(fontsize=7, loc="upper left")
        txt = fig.text(0.5, 0.96, "", ha="center", fontsize=10)
        imp = fig.text(0.5, 0.90, "", ha="center", fontsize=11, color="#d62d2d", weight="bold")
        fig.suptitle(title, x=0.02, ha="left", fontsize=9, y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        w, h = fig.canvas.get_width_height()
        path.parent.mkdir(parents=True, exist_ok=True)
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        n_trail = int(TRAIL_S / dt)
        for f in frames:
            for a, ln, keys in arts:
                if keys == [f"toe{kick}"]:
                    seg = P[keys[0]][max(0, f - n_trail):f + 1]
                    pts = seg
                else:
                    pts = np.vstack([P[k][f] for k in keys])
                if a is a3:
                    ln.set_data(pts[:, 0], pts[:, 1]); ln.set_3d_properties(pts[:, 2])
                elif a is a1:
                    ln.set_data(pts[:, 0], pts[:, 2])
                else:
                    ln.set_data(pts[:, 0], pts[:, 1])
            txt.set_text(f"{T('time from impact')} {tau[f]:+.2f} s   ({T('slow motion')} ×{speed:g})")
            imp.set_text(T("IMPACT") if abs(tau[f]) <= dt * 1.5 else "")
            fig.canvas.draw()
            vw.write(cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR))
        vw.release()
        plt.close(fig)
    return path


def animate_trial(trial: str, kick: str, speed: float = 0.2, out_dir: Path | None = None, langs=("en", "ja"), log=print) -> list[Path]:
    d = out_dir or OUT / trial
    f = d / "kinematics3d.csv"
    if not f.exists():
        log(f"[{trial}] kinematics3d.csv がありません(先に run.py imu3d)")
        return []
    df = pd.read_csv(f)
    out = []
    for lang in langs:
        if lang == "ja" and (not i18n.enabled() or i18n.japanese_font() is None):
            continue
        p = d / ("animation_ja.mp4" if lang == "ja" else "animation.mp4")
        out.append(render(df, kick, f"{trial} ({kick})", p, speed=speed, ja=lang == "ja"))
    log(f"[{trial}] " + ", ".join(p.name for p in out))
    return out


def run_animate(trials: list[str] | None = None, speed: float = 0.2, log=print) -> int:
    from .trials import load_trials

    tr = load_trials()
    names = trials or [n for n in tr if (OUT / n / "kinematics3d.csv").exists()]
    n = 0
    for name in names:
        if name not in tr:
            log(f"[スキップ] trials.csv に無い試行: {name}")
            continue
        try:
            n += len(animate_trial(name, tr[name].kick, speed, log=log))
        except Exception as e:  # noqa: BLE001  1試行の失敗で全体を止めない
            log(f"[失敗] {name}: {type(e).__name__}: {e}")
    log(f"アニメーション {n} 本")
    return n
