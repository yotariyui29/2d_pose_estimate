"""試行間の依存(`sequential`・`mediation`)の結果を、直感的に読める図にまとめる。  python run.py trialdep-figures

入力: config/landing.csv と、output/analysis/tables の sequential_acf(.csv / _detrend.csv)・mediation.csv。
出力: output/analysis/landing/
  trialdep_paths.png   被験者ごとの着弾を試行順に矢印でつないだ図(ジグザグ = 補正の繰り返し)
  trialdep_lag.png     「今回のずれ」と「次回のずれ」の散布図(傾き0 = 無関係、-1 = 完全に反対へ戻す)
  trialdep_summary.png ACF(1)(元と傾向除去後)の帰無範囲つきの図と、媒介解析の p 値の QQ プロット(対角線上 = 偶然の範囲)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta

from .config import ROOT
from .figio import save_figure
from .landing import GOAL_CENTER_Z, GOAL_H, GOAL_HALF_W, GROUND_Z
from .sequential import COLS, _perm_idx, load_series

READ_LAG = "Slope 0 = unrelated to the previous kick / -1 = reverses the previous error completely / + = stays near the previous one"


def plot_paths(series: list[dict], acf: pd.DataFrame | None, out: Path) -> None:
    """着弾を試行順に矢印でつなぐ(色 = 試行の順。早い = 暗い、遅い = 明るい)。"""
    n = len(series)
    fig, axs = plt.subplots(1, n, figsize=(4.3 * n, 3.6), sharex=True, sharey=True, squeeze=False)
    cmap = plt.get_cmap("viridis")
    for ax, s in zip(axs[0], series):
        y, z, m = s["y"], s["z"], len(s["y"])
        ax.add_patch(plt.Rectangle((-GOAL_HALF_W, GROUND_Z), 2 * GOAL_HALF_W, GOAL_H, fill=False, ec="#1a1a19", lw=1.2))
        ax.plot(0, GOAL_CENTER_Z, "x", color="#e0301e", ms=8, mew=2)
        for i in range(m - 1):
            ax.annotate("", xy=(y[i + 1], z[i + 1]), xytext=(y[i], z[i]),
                        arrowprops=dict(arrowstyle="->", color=cmap(0.9 * i / max(m - 2, 1)), lw=1.2, shrinkA=3, shrinkB=3))
        ax.scatter(y, z, c=np.arange(m), cmap=cmap, s=26, zorder=3, vmin=0, vmax=m - 1)
        for i in range(m):
            ax.annotate(str(int(s["no"][i])), (y[i], z[i]), textcoords="offset points", xytext=(4, 4), fontsize=7)
        sub = ""
        if acf is not None:
            r = acf[(acf["group"] == s["session"])].set_index("axis")
            if {"y", "z"} <= set(r.index):
                sub = f"\nACF(1) y {r.loc['y', 'acf1']:+.2f}, z {r.loc['z', 'acf1']:+.2f}"
        ax.set_title(f"{s['sid']} ({s['session']}), n={m}{sub}", fontsize=9)
        ax.set_xlim(-3.9, 3.9)
        ax.set_ylim(-0.5, 2.7)
        ax.set_aspect("equal")
        ax.set_xlabel("Horizontal y, kicking-leg side + (m)")
        ax.grid(alpha=0.25)
    axs[0, 0].set_ylabel("Vertical z (m)")
    fig.suptitle("Landing points joined in trial order (number = trial, dark = early, bright = late, x = target)", fontsize=10)
    fig.tight_layout()
    save_figure(fig, out, dpi=150)
    plt.close(fig)


def _pooled_slope(series: list[dict], key: str, rng: np.random.Generator, n_perm: int = 5000) -> tuple[float, np.ndarray]:
    """被験者の平均を引いた系列を全員まとめて e(n+1) = b e(n) の傾き b。帰無 = 被験者内で試行順をシャッフル。"""
    xs = [s[key] - s[key].mean() for s in series]
    num = sum((x[1:] * x[:-1]).sum() for x in xs)
    den = sum((x[:-1] ** 2).sum() for x in xs)
    nn, dd = np.zeros(n_perm), np.zeros(n_perm)
    for x in xs:
        xp = x[_perm_idx(len(x), n_perm, rng)]
        nn += (xp[:, 1:] * xp[:, :-1]).sum(1)
        dd += (xp[:, :-1] ** 2).sum(1)
    return float(num / den), nn / dd


def plot_lag(series: list[dict], out: Path, seed: int = 0) -> None:
    """今回のずれ(横)と次回のずれ(縦)。点が右下がり = 補正している。"""
    rng = np.random.default_rng(seed)
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, (key, title) in zip(axs, (("y", "Horizontal y (kicking-leg side +)"), ("z", "Vertical z"))):
        lim = 0
        for c, s in enumerate(series):
            x = s[key] - s[key].mean()
            ax.scatter(x[:-1], x[1:], s=34, color=COLS[c % len(COLS)], alpha=0.85, label=f"{s['sid']} ({s['session']})")
            lim = max(lim, np.abs(x).max())
        lim *= 1.1
        g = np.array([-lim, lim])
        b, null = _pooled_slope(series, key, rng)
        lo, hi = np.percentile(null, [2.5, 97.5])
        ax.plot(g, 0 * g, color="#6b6b66", ls="--", lw=1, label="slope 0: unrelated")
        ax.plot(g, -g, color="#6b6b66", ls=":", lw=1, label="slope -1: full reversal")
        ax.plot(g, b * g, color="#e0301e", lw=2.2, label=f"all subjects: slope {b:+.2f}")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("Error of this kick (m, mean removed)")
        ax.set_ylabel("Error of the next kick (m, mean removed)")
        ax.set_title(title + "\n" + f"slope {b:+.2f}  (shuffled order: 95% in [{lo:+.2f}, {hi:+.2f}])", fontsize=10)
        ax.grid(alpha=0.25)
    axs[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(READ_LAG, fontsize=9)
    fig.tight_layout()
    save_figure(fig, out, dpi=150)
    plt.close(fig)


def plot_summary(acf_raw: pd.DataFrame, acf_det: pd.DataFrame | None, med: pd.DataFrame | None, out: Path) -> None:
    """ACF(1)(元・傾向除去後)を帰無範囲つきで、媒介解析の p 値を QQ プロットで。"""
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    groups = list(dict.fromkeys(acf_raw["group"]))
    for ax, (key, title) in zip(axs[:2], (("y", "Horizontal y (kicking-leg side +)"), ("z", "Vertical z"))):
        for i, g in enumerate(groups):
            for off, df, col, mk in ((-0.17, acf_raw, "#1a1a19", "o"), (0.17, acf_det, "#eb6834", "D")):
                if df is None:
                    continue
                r = df[(df["group"] == g) & (df["axis"] == key)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.plot([i + off] * 2, [r["null_lo"], r["null_hi"]], color="#b7b7b0", lw=6, solid_capstyle="butt")
                ax.plot(i + off, r["acf1"], mk, color=col, ms=6, zorder=3, mec="white" if mk == "D" else col)
        ax.axhline(0, color="#6b6b66", lw=0.8)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g if g in ("ALL", "EXP") else acf_raw[acf_raw["group"] == g]["subject_id"].iloc[0] for g in groups], fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.95, 0.75)
        ax.grid(alpha=0.25, axis="y")
    axs[0].set_ylabel("Lag-1 autocorrelation")
    axs[0].plot([], [], color="#b7b7b0", lw=6, label="95% range of shuffled order")
    axs[0].plot([], [], "o", color="#1a1a19", label="original")
    axs[0].plot([], [], "D", color="#eb6834", label="linear trend removed")
    axs[0].legend(fontsize=8, loc="upper right")
    ax = axs[2]
    if med is not None and not med.empty:
        d = med[med["group"].isin(["ALL", "EXP"]) & (med["n_subj"] >= 2)]
        p = np.sort(d["p_ind"].dropna().to_numpy())
        m = len(p)
        if m:
            exp = (np.arange(1, m + 1) - 0.5) / m
            ax.fill_between(exp, beta.ppf(0.025, np.arange(1, m + 1), m - np.arange(1, m + 1) + 1),
                            beta.ppf(0.975, np.arange(1, m + 1), m - np.arange(1, m + 1) + 1), color="#e1e1dc", label="95% range if chance")
            ax.plot([0, 1], [0, 1], color="#6b6b66", lw=1)
            ax.plot(exp, p, "o", color="#2a78d6", ms=3.5, label=f"{m} paths (knob x error pair)")
            ax.text(0.04, 0.9, f"smallest corrected p = {d['p_ind_adj'].min():.2f}", transform=ax.transAxes, fontsize=9)
    ax.set_xlabel("Expected p-value if only chance")
    ax.text(0.04, 0.82, "paths are correlated: the band is a guide", transform=ax.transAxes, fontsize=8, color="#6b6b66")
    ax.set_ylabel("Observed p-value, indirect effect via knob")
    ax.set_title("Which knob carries the correction? (on the diagonal = chance)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, out, dpi=150)
    plt.close(fig)


def run_trialdep_figures(root: Path | None = None, log=print) -> None:
    root = root or ROOT / "output"
    tdir, fdir = root / "analysis" / "tables", root / "analysis" / "landing"

    def rd(name: str) -> pd.DataFrame | None:
        p = tdir / name
        return pd.read_csv(p) if p.exists() else None

    series = load_series()
    acf_raw = rd("sequential_acf.csv")
    if not series or acf_raw is None:
        raise SystemExit("先に `python run.py sequential` を実行してください(sequential_acf.csv が必要です)")
    plot_paths(series, acf_raw, fdir / "trialdep_paths.png")
    plot_lag(series, fdir / "trialdep_lag.png")
    plot_summary(acf_raw, rd("sequential_acf_detrend.csv"), rd("mediation.csv"), fdir / "trialdep_summary.png")
    log(f"保存: {fdir}(trialdep_paths.png / trialdep_lag.png / trialdep_summary.png と日本語版)")
