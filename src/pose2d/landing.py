"""着弾位置(ゴール平面)の図と分布の楕円・傾きの算出。  python run.py landing [--session S]

入力: config/landing.csv (session,trial,y_m,z_m[,source])。ボール中心が原点、ゴール中心 (11,0,1)。
      y>0 は被験者から見て右側(図の横軸は右が +)。蹴り足側が + になる y は kick_side から換算する(右足はそのまま、左足は符号反転)。
出力: output/analysis/landing/<session>.png  と  output/analysis/tables/landing_summary.csv
図は発射角度のズレ 1,2,3,5,7,10,15° の等角線(ボール→ゴール中心ベクトルとのなす角)・ゴール枠・試行番号・95%楕円。
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ROOT
from .figio import save_figure
from .kickfactors import load_landing, tilt_deg
from .sessions import list_sessions, load_session

GOAL_X, GOAL_CENTER_Z = 11.0, 1.0
GOAL_HALF_W, GOAL_H, GROUND_Z = 3.66, 2.44, -0.22  # ゴール下端=-0.22 m(ボール半径の2倍下)
ANGLES = [(1, "#7b2d26"), (2, "#e0301e"), (3, "#f08c00"), (5, "#e8c200"), (7, "#2f9e44"), (10, "#1c64f2"), (15, "#8e24aa")]
CHI2_95 = 5.991  # 2自由度のカイ二乗(95%)


def ellipse(y: np.ndarray, z: np.ndarray, chi2: float = CHI2_95) -> dict:
    """共分散楕円(中心, 長軸/短軸の半径, 長軸の傾き)。傾き: 長軸が y 軸となす角(+ = y が大きいほど z も大きい)。"""
    c = np.cov(np.vstack([y, z]))
    w, V = np.linalg.eigh(c)
    order = np.argsort(w)[::-1]
    w, V = np.maximum(w[order], 0), V[:, order]
    return dict(cy=float(y.mean()), cz=float(z.mean()), a=float(np.sqrt(chi2 * w[0])), b=float(np.sqrt(chi2 * w[1])),
                tilt_deg=tilt_deg(y, z), sd_y=float(y.std(ddof=1)), sd_z=float(z.std(ddof=1)),
                corr_yz=float(np.corrcoef(y, z)[0, 1]))


def _angle_field(yy: np.ndarray, zz: np.ndarray) -> np.ndarray:
    d = np.stack([np.full_like(yy, GOAL_X), yy, zz], -1)
    a = np.array([GOAL_X, 0.0, GOAL_CENTER_Z])
    cos = (d @ a) / (np.linalg.norm(d, axis=-1) * np.linalg.norm(a))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def plot_landing(df: pd.DataFrame, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.4))
    yy, zz = np.meshgrid(np.linspace(-4, 4, 400), np.linspace(-0.6, 2.6, 300))
    ang = _angle_field(yy, zz)
    for deg, col in ANGLES:
        cs = ax.contour(yy, zz, ang, levels=[deg], colors=[col], linewidths=1.2)
        ax.clabel(cs, fmt=lambda v: f"{v:.0f}°", fontsize=7)
    ax.add_patch(plt.Rectangle((-GOAL_HALF_W, GROUND_Z), 2 * GOAL_HALF_W, GOAL_H, fill=False, ec="#1a1a19", lw=2))
    ax.plot(0, GOAL_CENTER_Z, "x", color="#e0301e", ms=10, mew=3)
    ax.scatter(df["y_m"], df["z_m"], s=420, c="#1a1a19", zorder=3)
    for r in df.itertuples():
        no = re.search(r"(\d+)$", str(r.trial))
        ax.text(r.y_m, r.z_m, no[1] if no else "?", color="white", ha="center", va="center", fontsize=10, zorder=4, weight="bold")
    if len(df) >= 3:
        e = ellipse(df["y_m"].to_numpy(), df["z_m"].to_numpy())
        # 楕円は主軸の向きで描く(傾き=長軸と y 軸のなす角)
        t = np.linspace(0, 2 * np.pi, 200)
        th = np.radians(e["tilt_deg"])
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        xy = R @ np.vstack([e["a"] * np.cos(t), e["b"] * np.sin(t)])
        ax.plot(e["cy"] + xy[0], e["cz"] + xy[1], color="#d62d6e", lw=2, zorder=5, label=f"95% ellipse (tilt in this plot {e['tilt_deg']:+.0f}°)")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-0.6, 2.6)
    ax.set_aspect("equal")
    ax.set_xlabel("Horizontal y (m)")
    ax.set_ylabel("Vertical z (m)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, out, dpi=150)
    plt.close(fig)


def run_landing(sessions: list[str] | None = None, root: Path | None = None, log=print) -> pd.DataFrame:
    root = root or ROOT / "output"
    land = load_landing()
    if land.empty:
        raise SystemExit("config/landing.csv がありません(列: session,trial,y_m,z_m)")
    rows, pts = [], {}
    for sess in sessions or list(dict.fromkeys(land["session"])):
        g = land[land["session"] == sess].dropna(subset=["y_m", "z_m"])
        if g.empty:
            continue
        side = (_first(g, "kick_side") or "").lower() or (load_session(sess).kick_side if sess in list_sessions() else "right")
        sid = _first(g, "subject_id") or sess
        exp = pd.to_numeric(g["experience_y"], errors="coerce").iloc[0] if "experience_y" in g else np.nan
        plot_landing(g, f"{sid} ({side}-footed{f', {exp:g} y' if np.isfinite(exp) else ''}), n={len(g)}",
                     root / "analysis" / "landing" / f"{sess}.png")
        sgn = 1.0 if side == "right" else -1.0  # y>0 は被験者から見て右側 → 右足はそのまま、左足は反転
        row = dict(session=sess, subject_id=sid, experience_y=exp, kick_side=side, n=len(g),
                   mean_y=g["y_m"].mean(), mean_z=g["z_m"].mean())
        pts[sid] = (sgn * g["y_m"].to_numpy(), g["z_m"].to_numpy(), exp)
        if len(g) >= 3:
            e = ellipse(sgn * g["y_m"].to_numpy(), g["z_m"].to_numpy())
            row.update(ellipse_a95_m=e["a"], ellipse_b95_m=e["b"], tilt_kickside_deg=e["tilt_deg"],
                       sd_y_kickside=e["sd_y"], sd_z=e["sd_z"], corr_kickside_y_z=e["corr_yz"])
            log(f"[{sess}] n={len(g)} 長軸の傾き(蹴り足側y基準) {e['tilt_deg']:+.1f}°  r(y,z)={e['corr_yz']:+.2f}")
        rows.append(row)
    df = pd.DataFrame(rows)
    out = root / "analysis" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "landing_summary.csv", index=False, float_format="%.4f")
    if len(pts) > 1:
        plot_all(pts, root / "analysis" / "landing" / "all_subjects_kickside.png")
    log(f"保存: {out / 'landing_summary.csv'} / {root / 'analysis' / 'landing'}")
    return df


def _first(g: pd.DataFrame, col: str) -> str | None:
    v = g[col].dropna() if col in g else []
    return str(v.iloc[0]).strip() if len(v) else None


def plot_all(pts: dict, out: Path) -> None:
    """全被験者を蹴り足側 y が + になるよう揃えて重ね、被験者ごとの95%楕円(長軸の傾き)を描く。"""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6b6b66"]
    for i, (sid, (y, z, exp)) in enumerate(pts.items()):
        c = cols[i % len(cols)]
        ax.scatter(y, z, s=22, color=c, alpha=0.8)
        if len(y) >= 3:
            e = ellipse(y, z)
            th, t = np.radians(e["tilt_deg"]), np.linspace(0, 2 * np.pi, 200)
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            xy = R @ np.vstack([e["a"] * np.cos(t), e["b"] * np.sin(t)])
            lab = f"{sid}" + (f" ({exp:g}y)" if np.isfinite(exp) else "") + f"  tilt {e['tilt_deg']:+.0f}°"
            ax.plot(e["cy"] + xy[0], e["cz"] + xy[1], color=c, lw=1.8, label=lab)
    ax.add_patch(plt.Rectangle((-GOAL_HALF_W, GROUND_Z), 2 * GOAL_HALF_W, GOAL_H, fill=False, ec="#1a1a19", lw=1.5))
    ax.plot(0, GOAL_CENTER_Z, "x", color="#e0301e", ms=9, mew=2.5)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-0.6, 2.6)
    ax.set_aspect("equal")
    ax.set_xlabel("Horizontal y, kicking-leg side + (m)  [left-footed subjects mirrored]")
    ax.set_ylabel("Vertical z (m)")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, out, dpi=150)
    plt.close(fig)
