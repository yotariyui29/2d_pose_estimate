"""全試行の集計・表・図(被験者ごと/被験者間)。  python run.py analyze

入力: output/<session>/<trial>/{kinematics.csv, summary.json}  (run の出力)
出力: output/analysis/
  tables/  metrics.csv(全指標) ball_speed_table.csv ball_speed_by_subject.csv report.md
  subjects/<subject>/  fig3_momentum_shank.png fig4_segment_energy.png effective_mass.png extra.png
  compare/             fig3.. fig5_boxplots.png effective_mass.png extra.png  (被験者間比較)
図の対応(SI2025): Fig.3=全身運動量+蹴り脚下腿角速度, Fig.4=セグメントエネルギー, Fig.5=E_b/W_p/EIの箱ひげ。
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ROOT
from .figio import save_figure
from .mechanics2d import analyze_trial
from .sessions import list_sessions

warnings.filterwarnings("ignore", category=RuntimeWarning)  # n=1の標準偏差など

# カテゴリ色は被験者(エンティティ)に固定。表示する人数が変わっても色は変えない。
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
LINES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (1, 1))]
MARKERS = ["o", "s", "^", "D", "v", "P"]
INK, MUTED, GRID_C = "#1a1a19", "#6b6b66", "#e3e2dc"
T_GRID = np.arange(-0.3, 0.1 + 1e-9, 0.0005)
XLIM = (-0.2, 0.1)

SEG_PANELS = [("kick_torso", "Kicking leg torso"), ("kick_thigh", "Kicking leg thigh"), ("kick_shank", "Kicking leg shank"),
              ("support_torso", "Support leg torso"), ("support_thigh", "Support leg thigh"), ("support_shank", "Support leg shank")]


@dataclass
class Trial:
    session: str
    subject: str
    name: str
    metrics: dict
    ts: pd.DataFrame  # T_GRID上に補間した時系列(列: t, momentum_x, ...)
    info: dict = field(default_factory=dict)


# ---------- 読み込み・集計 ----------
def _resample(ts: pd.DataFrame) -> pd.DataFrame:
    out = {"t": T_GRID}
    t = ts["t"].to_numpy()
    for c in ts.columns:
        if c == "t":
            continue
        y = ts[c].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(t)
        out[c] = np.interp(T_GRID, t[ok], y[ok], left=np.nan, right=np.nan) if ok.sum() > 3 else np.full(len(T_GRID), np.nan)
    return pd.DataFrame(out)


def load_trials(root: Path, sessions: list[str] | None = None, log=print) -> list[Trial]:
    order = list_sessions()
    trials: list[Trial] = []
    for sess in (sessions or order):
        for kin in sorted((root / sess).glob("*/kinematics.csv")):
            sj = kin.parent / "summary.json"
            if not sj.exists():
                continue
            info = json.loads(sj.read_text(encoding="utf-8"))
            k = pd.read_csv(kin)
            if "t_rel_impact_s" not in k:
                log(f"[スキップ] {kin.parent.name}: インパクト未設定")
                continue
            try:
                metrics, ts = analyze_trial(k, info)
            except Exception as e:  # 1試行の失敗で全体を止めない
                log(f"[スキップ] {kin.parent.name}: {type(e).__name__}: {e}")
                continue
            trials.append(Trial(sess, info.get("subject", sess), kin.parent.name, metrics, _resample(ts), info))
    return trials


def metrics_table(trials: list[Trial]) -> pd.DataFrame:
    rows = [{"session": t.session, "subject": t.subject, "trial": t.name, **t.metrics} for t in trials]
    return pd.DataFrame(rows)


def ball_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["session", "subject", "trial", "ball_speed_mps", "ball_speed_kmh", "E_b_J", "toe_speed_impact_mps", "ball_foot_ratio"]
    per_trial = df[cols].copy()
    g = df.groupby("subject", sort=False)
    by_subj = pd.DataFrame({
        "n_trials": g.size(), "n_with_ball": g["ball_speed_mps"].count(),
        "ball_speed_mean_mps": g["ball_speed_mps"].mean(), "ball_speed_sd_mps": g["ball_speed_mps"].std(),
        "ball_speed_mean_kmh": g["ball_speed_kmh"].mean(), "ball_speed_sd_kmh": g["ball_speed_kmh"].std(),
        "ball_speed_max_kmh": g["ball_speed_kmh"].max(),
    }).reset_index()
    return per_trial, by_subj


def _md(df: pd.DataFrame, fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    body = ""
    for _, r in df.iterrows():
        cells = [("" if pd.isna(v) else (format(v, fmt[c]) if c in fmt and not isinstance(v, str) else (f"{v:.2f}" if isinstance(v, float) else str(v))))
                 for c, v in r.items()]
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


# ---------- 描画ヘルパ ----------
def _style() -> None:
    plt.rcParams.update({
        "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID_C,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white", "axes.titlesize": 12, "axes.titleweight": "bold",
    })


def _stack(trials: list[Trial], col: str) -> np.ndarray:
    return np.vstack([t.ts[col].to_numpy() for t in trials])


def _mean_sd(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(all="ignore"):
        n = np.isfinite(a).sum(0)
        m = np.where(n > 0, np.nansum(a, 0) / np.maximum(n, 1), np.nan)
        sd = np.where(n > 1, np.nanstd(a, 0, ddof=1), np.nan)
    return m, sd


def _by_subject(trials: list[Trial]) -> dict[str, list[Trial]]:
    d: dict[str, list[Trial]] = {}
    for t in trials:
        d.setdefault(t.subject, []).append(t)
    return d


def _style_of(subject: str, subjects: list[str]) -> dict:
    i = subjects.index(subject)
    return dict(color=PALETTE[i % len(PALETTE)], ls=LINES[i % len(LINES)])


def _finish_time_axis(ax, ylabel: str, title: str | None = None, zero=True) -> None:
    ax.set_xlim(*XLIM)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left")
    if zero:
        ax.axvline(0, color=MUTED, lw=1, ls=":")


def _individual(ax, trs: list[Trial], col: str, style: dict, scale: float = 1.0) -> None:
    """被験者1人: 各試行を細線、平均を太線。"""
    for t in trs:
        ax.plot(T_GRID, t.ts[col] * scale, color=style["color"], lw=0.8, alpha=0.35)
    m, _ = _mean_sd(_stack(trs, col) * scale)
    ax.plot(T_GRID, m, color=style["color"], lw=2.4, ls="-", label=f"mean (n={len(trs)})")


def _compare(ax, groups: dict[str, list[Trial]], col: str, subjects: list[str], scale: float = 1.0, band: bool = True) -> None:
    for s, trs in groups.items():
        st = _style_of(s, subjects)
        m, sd = _mean_sd(_stack(trs, col) * scale)
        ax.plot(T_GRID, m, color=st["color"], ls=st["ls"], lw=2.0, label=f"{s} (n={len(trs)})")
        if band:
            ax.fill_between(T_GRID, m - sd, m + sd, color=st["color"], alpha=0.12, lw=0)


def _save(fig, path: Path) -> None:
    save_figure(fig, path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _legend_fig(fig, ax) -> None:
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=min(len(lab), 6), bbox_to_anchor=(0.5, -0.02))


# ---------- 被験者ごとの図 ----------
def subject_figures(subject: str, trs: list[Trial], subjects: list[str], out: Path) -> None:
    st = _style_of(subject, subjects)
    d = out / "subjects" / subject

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    _individual(ax[0], trs, "momentum_x", st)
    _finish_time_axis(ax[0], "Momentum (kg m/s)", "(A) Whole-body momentum (forward)")
    _individual(ax[1], trs, "omega_shank", st)
    _finish_time_axis(ax[1], "Angular velocity (rad/s)", "(B) Kicking-leg shank angular velocity")
    for a in ax:
        a.set_xlabel("Time from impact (s)")
    ax[0].legend(loc="best")
    fig.suptitle(f"{subject}: thin = trials, thick = mean", color=MUTED, fontsize=10, y=1.02)
    _save(fig, d / "fig3_momentum_shank.png")

    fig, axs = plt.subplots(2, 3, figsize=(13, 6.6), sharex=True)
    for a, (key, title) in zip(axs.ravel(), SEG_PANELS):
        _individual(a, trs, f"E_{key}", st)
        _finish_time_axis(a, "Segment energy (J)", title)
    for a in axs[1]:
        a.set_xlabel("Time from impact (s)")
    axs[0, 0].legend(loc="best")
    fig.suptitle(f"{subject}: segment kinetic energy (translation + rotation)", color=MUTED, fontsize=10)
    fig.tight_layout()
    _save(fig, d / "fig4_segment_energy.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    _individual(ax, trs, "m_eff", st)
    _finish_time_axis(ax, "Effective mass of kicking foot (kg)", f"{subject}: effective mass (planar 3-link leg, Khatib)")
    ax.set_xlabel("Time from impact (s)")
    ax.set_xlim(-0.12, 0.05)
    ax.set_ylim(0, None)
    ax.legend(loc="best")
    _save(fig, d / "effective_mass.png")

    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8))
    _individual(axs[0, 0], trs, "toe_speed", st)
    _finish_time_axis(axs[0, 0], "Speed (m/s)", "(A) Kicking-foot toe speed")
    _individual(axs[0, 1], trs, "knee_flexion", st)
    _finish_time_axis(axs[0, 1], "Angle (deg)", "(B) Kicking-leg knee flexion")
    for n, c, lab in (("thigh", PALETTE[0], "thigh"), ("shank", PALETTE[1], "shank"), ("foot", PALETTE[2], "foot")):
        m, _ = _mean_sd(_stack(trs, f"omega_{n}"))
        axs[1, 0].plot(T_GRID, m, color=c, lw=2.2, label=lab)
    _finish_time_axis(axs[1, 0], "Angular velocity (rad/s)", "(C) Segment angular velocities (mean)")
    axs[1, 0].legend(loc="best")
    for a in (axs[0, 0], axs[0, 1], axs[1, 0]):
        a.set_xlabel("Time from impact (s)")
    names = [t.name.replace(f"{subject}_", "") for t in trs]
    v = [t.metrics["ball_speed_kmh"] for t in trs]
    axs[1, 1].bar(range(len(trs)), v, color=st["color"], width=0.65)
    axs[1, 1].set_xticks(range(len(trs)), names, rotation=45, ha="right", fontsize=9)
    axs[1, 1].set_ylabel("Ball speed (km/h)")
    axs[1, 1].set_title("(D) Ball speed per trial", loc="left")
    axs[1, 1].grid(axis="x", visible=False)
    fig.suptitle(subject, color=MUTED, fontsize=10)
    fig.tight_layout()
    _save(fig, d / "extra.png")


# ---------- 被験者間比較 ----------
def _box(ax, groups: dict[str, np.ndarray], subjects: list[str], ylabel: str, title: str) -> None:
    names = list(groups)
    data = [groups[n][np.isfinite(groups[n])] for n in names]
    pos = np.arange(len(names))
    bp = ax.boxplot(data, positions=pos, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.6), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
    rng = np.random.default_rng(0)
    for i, (n, box) in enumerate(zip(names, bp["boxes"])):
        c = PALETTE[subjects.index(n) % len(PALETTE)]
        box.set(facecolor=c, alpha=0.28, edgecolor=c, lw=1.4)
        ax.scatter(pos[i] + rng.uniform(-0.12, 0.12, len(data[i])), data[i], s=26, color=c, edgecolor="white", lw=0.8,
                   marker=MARKERS[subjects.index(n) % len(MARKERS)], zorder=3)
    ax.set_xticks(pos, names, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="x", visible=False)


def compare_figures(trials: list[Trial], df: pd.DataFrame, subjects: list[str], out: Path) -> None:
    g = _by_subject(trials)
    d = out / "compare"

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    _compare(ax[0], g, "momentum_x", subjects)
    _finish_time_axis(ax[0], "Momentum (kg m/s)", "(A) Whole-body momentum (forward)")
    _compare(ax[1], g, "omega_shank", subjects)
    _finish_time_axis(ax[1], "Angular velocity (rad/s)", "(B) Kicking-leg shank angular velocity")
    for a in ax:
        a.set_xlabel("Time from impact (s)")
    _legend_fig(fig, ax[0])
    fig.suptitle("Subject comparison: mean ± SD over trials", color=MUTED, fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    _save(fig, d / "fig3_momentum_shank.png")

    fig, axs = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for a, (key, title) in zip(axs.ravel(), SEG_PANELS):
        _compare(a, g, f"E_{key}", subjects, band=False)
        _finish_time_axis(a, "Segment energy (J)", title)
    for a in axs[1]:
        a.set_xlabel("Time from impact (s)")
    _legend_fig(fig, axs[0, 0])
    fig.suptitle("Segment kinetic energy: subject means", color=MUTED, fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, d / "fig4_segment_energy.png")

    fig, axs = plt.subplots(1, 4, figsize=(16, 4.4))
    for a, (col, lab, title) in zip(axs, [("ball_speed_mps", "Ball speed (m/s)", "Ball speed"),
                                         ("E_b_J", "$E_b$ (J)", "Effective energy $E_b$"),
                                         ("W_p_J", "$W_p$ (J)", "Mechanical work $W_p$"),
                                         ("EI", "EI ($E_b/W_p$)", "Effectiveness index EI")]):
        _box(a, {s: df.loc[df["subject"] == s, col].to_numpy(float) for s in g}, subjects, lab, title)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.5, 0.01, "$E_b$ = 1/2 m v² (v: ball speed 0.01-0.02 s after impact, m = 0.43 kg);  "
             "$W_p$ = kicking-leg sum of |dE| over the 0.1 s before impact;  EI = $E_b$/$W_p$",
             ha="center", color=MUTED, fontsize=9)
    _save(fig, d / "fig5_boxplots.png")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1.6, 1, 1]))
    _compare(axs[0], g, "m_eff", subjects)
    _finish_time_axis(axs[0], "Effective mass (kg)", "(A) Kicking-foot effective mass")
    axs[0].set_xlim(-0.12, 0.05)
    axs[0].set_xlabel("Time from impact (s)")
    axs[0].set_ylim(0, None)
    for a, col, title in ((axs[1], "m_eff_impact_kg", "(B) At impact"), (axs[2], "m_eff_pre10ms_kg", "(C) Mean of last 10 ms")):
        _box(a, {s: df.loc[df["subject"] == s, col].to_numpy(float) for s in g}, subjects, "Effective mass (kg)", title)
    _legend_fig(fig, axs[0])
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, d / "effective_mass.png")

    fig, axs = plt.subplots(2, 3, figsize=(15, 8.4))
    _compare(axs[0, 0], g, "toe_speed", subjects, band=False)
    _finish_time_axis(axs[0, 0], "Speed (m/s)", "(A) Kicking-foot toe speed")
    _compare(axs[0, 1], g, "knee_flexion", subjects, band=False)
    _finish_time_axis(axs[0, 1], "Angle (deg)", "(B) Kicking-leg knee flexion")
    for a in axs[0, :2]:
        a.set_xlabel("Time from impact (s)")
    ax = axs[0, 2]  # 近位→遠位の角速度ピーク時刻
    for i, n in enumerate(("thigh", "shank", "foot")):
        for s in g:
            v = df.loc[df["subject"] == s, f"peak_t_{n}_s"].to_numpy(float) * 1000
            if np.isfinite(v).any():
                st = _style_of(s, subjects)
                j = list(g).index(s)
                ax.scatter(np.nanmean(v), i + (j - (len(g) - 1) / 2) * 0.12, color=st["color"], s=55,
                           marker=MARKERS[subjects.index(s) % len(MARKERS)], zorder=3)
                ax.errorbar(np.nanmean(v), i + (j - (len(g) - 1) / 2) * 0.12, xerr=np.nanstd(v), color=st["color"], lw=1.2, capsize=2)
    ax.set_yticks([0, 1, 2], ["thigh", "shank", "foot"])
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, lw=1, ls=":")
    ax.set_xlabel("Time of peak angular velocity from impact (ms)")
    ax.set_title("(C) Proximal-to-distal sequence", loc="left")
    ax = axs[1, 0]
    x, y = df["toe_speed_impact_mps"], df["ball_speed_mps"]
    for s in g:
        m = df["subject"] == s
        ax.scatter(x[m], y[m], color=_style_of(s, subjects)["color"], marker=MARKERS[subjects.index(s) % len(MARKERS)], s=36,
                   edgecolor="white", lw=0.8)
    lim = np.nanmax(np.r_[x, y]) * 1.05 if np.isfinite(np.r_[x, y]).any() else 30
    for r in (1.0, 1.2, 1.4):
        ax.plot([0, lim / r], [0, lim], color=MUTED, lw=0.8, ls=":")
        ax.text(lim / r * 0.98, lim * 0.98, f"{r:.1f}", color=MUTED, fontsize=8, ha="right", va="top")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Toe speed at impact (m/s)")
    ax.set_ylabel("Ball speed (m/s)")
    ax.set_title("(D) Ball vs toe speed (dotted: ratio)", loc="left")
    _box(axs[1, 1], {s: df.loc[df["subject"] == s, "ball_foot_ratio"].to_numpy(float) for s in g}, subjects,
         "Ball speed / toe speed", "(E) Collision transfer ratio")
    ax = axs[1, 2]
    for s in g:
        m = df["subject"] == s
        ax.scatter(df.loc[m, "ball_speed_mps"], df.loc[m, "EI"], color=_style_of(s, subjects)["color"],
                   marker=MARKERS[subjects.index(s) % len(MARKERS)], s=36, edgecolor="white", lw=0.8)
    ax.set_xlabel("Ball speed (m/s)")
    ax.set_ylabel("EI")
    ax.set_title("(F) EI vs ball speed", loc="left")
    _legend_fig(fig, axs[0, 0])
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, d / "extra.png")


# ---------- エントリ ----------
def run_analysis(root: Path | None = None, out: Path | None = None, sessions: list[str] | None = None, log=print) -> Path:
    root = root or ROOT / "output"
    out = out or root / "analysis"
    _style()
    trials = load_trials(root, sessions, log)
    if not trials:
        raise SystemExit(f"解析対象がありません: {root}/<session>/<trial>/kinematics.csv")
    df = metrics_table(trials)
    subjects = list(dict.fromkeys(t.subject for t in trials))
    tdir = out / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tdir / "metrics.csv", index=False, float_format="%.5g")
    per_trial, by_subj = ball_tables(df)
    per_trial.to_csv(tdir / "ball_speed_table.csv", index=False, float_format="%.4g")
    by_subj.to_csv(tdir / "ball_speed_by_subject.csv", index=False, float_format="%.4g")
    (tdir / "report.md").write_text(
        "# ボール速度(インパクト後0.01〜0.02sの位置の直線近似、2次元平面内)\n\n## 被験者別\n\n"
        + _md(by_subj) + "\n## 試行別\n\n" + _md(per_trial.drop(columns="session")) + "\n", encoding="utf-8")
    for s, trs in _by_subject(trials).items():
        subject_figures(s, trs, subjects, out)
    compare_figures(trials, df, subjects, out)
    miss = int(df["ball_speed_mps"].isna().sum())
    log(f"{len(trials)} 試行 / {len(subjects)} 名 → {out}" + (f"  [ボール速度なし {miss} 試行]" if miss else ""))
    return out
