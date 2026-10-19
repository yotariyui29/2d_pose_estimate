"""前の試行のずれ → 次の試行の運動学(補正のつまみ) → 次の着弾。 python run.py mediation

`sequential`(着弾だけの試行間依存)の続き。補正が起きているなら、被験者は前の誤差を見て「何か」を変えているはずで、
その「何か」を要因表(`factors` の kick_factors.csv)の運動学の量から探す。経路(すべて被験者内で順位正規化 = 外れ値に強い):

  e_c(n) --a--> k(n+1) --b--> e_c'(n+1)        c, c' ∈ {y(左右, 蹴り足側+), z(上下)}、k = 運動学の量(つまみ)
  a: 前の誤差 e_c(n) が次の運動学 k(n+1) に及ぼす影響(**k(n) を一定にした**偏回帰係数 = 標準化)
  b: 次の運動学 k(n+1) と次の着弾 e_c'(n+1) の相関
  間接効果 = a × b(標準化。「前の誤差 → つまみ → 次の誤差」の経路の大きさ)、total = corr(e_c(n), e_c'(n+1))(`sequential` の K に近い)

- a で k(n) を一定にするのは、**平均への回帰**(k(n) が e(n) の原因で k が自分の平均へ戻るだけ)を補正と取り違えないため。
- 検定は被験者内で**試行順をシャッフル**した帰無分布(試行内の k と e の対応は保つ)。被験者をまとめた値(ALL、EXP = 8年以上)は
  被験者ごとの平均で、被験者ごとに独立にシャッフルする。つまみは多数あるので、グループの p は「つまみ全体の最大値(max-T)」で補正した値も出す。
- 試行は連続(n と n+1 が両方ある)の組だけ使う。欠損(Xsens の除外など)のある試行はその組を外す。

制約: 組の数は被験者あたり 5〜10 しかなく検出力は低い。k は測った量だけ(接触点など測れていない量が本命の可能性)。因果ではなく手がかり。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from .config import ROOT
from .figio import save_figure
from .kickfactors import load_landing

# 補正の「つまみ」の候補(kick_factors.csv の列)。着弾そのもの(ボール出射の仰角・速度など)は結果なので入れない
KNOBS = (
    "toe_vel_elev_deg", "launch_minus_toevel_deg", "contact_dx_m", "contact_dy_m", "foot_pitch_deg", "toe_speed_impact_mps",
    "com_minus_ball_x_m", "com_height_m", "hip_minus_ball_x_m", "support_ankle_minus_ball_x_m", "trunk_lean_fwd_deg",
    "thigh_fwd_deg", "shank_fwd_deg", "knee_flexion_impact_deg", "arc_phase_ms", "arc_phase_rise_m",
    "xs_support_thigh_lat_deg", "xs_support_thigh_fwd_deg", "xs_kick_shank_lat_deg", "xs_kick_thigh_lat_deg",
    "xs_pelvis_lean_kickside_deg", "xs_swing_axis_tilt_deg", "xs_thigh_peak_to_shank_peak_ms", "xs_kickleg_ratio",
)
MIN_PAIRS = 5  # 連続する試行の組がこれ未満の(被験者, つまみ)は対象外
EXP_YEARS = 8
COMBOS = (("y", "y"), ("z", "z"), ("y", "z"), ("z", "y"))  # (前の誤差の成分, 次の誤差の成分)


def rank_normal(x: np.ndarray) -> np.ndarray:
    """順位を正規スコアに(欠損は NaN のまま)。被験者内で使う。"""
    x = np.asarray(x, float)
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x)
    if ok.sum() >= 3:
        out[ok] = norm.ppf((rankdata(x[ok]) - 0.5) / ok.sum())
    return out


def _mcorr(A: np.ndarray, B: np.ndarray, M: np.ndarray) -> np.ndarray:
    """行ごとの相関(M が True の列だけ)。A, B, M は (行, 列)。"""
    Mf = M.astype(float)
    n = Mf.sum(1)
    A0, B0 = np.where(M, A, 0.0), np.where(M, B, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = (A0 - (A0.sum(1) / n)[:, None]) * Mf
        b = (B0 - (B0.sum(1) / n)[:, None]) * Mf
        r = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
    r[n < MIN_PAIRS] = np.nan
    return r


def path_stats(E: np.ndarray, k: np.ndarray) -> dict[str, np.ndarray]:
    """E: (行, n, 2) 順位正規化した着弾(y, z)を試行順に並べたもの。k: (行, n) つまみ(欠損は NaN)。行 = シャッフルの1回分。
    返す値: a (行, 2) = 前の誤差 c → 次のつまみ(k(n)一定の標準化偏回帰)、b (行, 2) = 次のつまみ → 次の誤差 c'、
    ind (行, 2, 2)[c, c'] = a_c × b_c'、total (行, 2, 2)[c, c'] = corr(e_c(n), e_c'(n+1))。"""
    v, w = k[:, :-1], k[:, 1:]
    M = np.isfinite(v) & np.isfinite(w)
    r_wv = _mcorr(w, v, M)
    a = np.empty((len(k), 2))
    b = np.empty((len(k), 2))
    for c in range(2):
        u = E[:, :-1, c]
        r_uv, r_wu = _mcorr(u, v, M), _mcorr(w, u, M)
        with np.errstate(invalid="ignore", divide="ignore"):
            a[:, c] = (r_wu - r_wv * r_uv) / (1 - r_uv ** 2)
        b[:, c] = _mcorr(w, E[:, 1:, c], M)
    allm = np.ones(E[:, :-1, 0].shape, bool)
    total = np.stack([np.stack([_mcorr(E[:, :-1, c], E[:, 1:, c2], allm) for c2 in range(2)], -1) for c in range(2)], 1)
    return dict(a=a, b=b, ind=a[:, :, None] * b[:, None, :], total=total)


def load_trials(sessions: list[str] | None = None, root: Path | None = None) -> list[dict]:
    """kick_factors.csv(着弾つき)から被験者ごとの試行系列(試行番号順)。"""
    root = root or ROOT / "output"
    p = root / "analysis" / "tables" / "kick_factors.csv"
    if not p.exists():
        raise SystemExit(f"{p} がありません。先に `python run.py factors`(または import-results)を実行してください")
    d = pd.read_csv(p)
    land = load_landing()
    out = []
    for sess, g in d.groupby("session"):
        if sessions and sess not in sessions:
            continue
        g = g.dropna(subset=["land_y_kickside_m", "land_z_m"]).sort_values("trial_no")
        if len(g) < MIN_PAIRS + 1:
            continue
        ls = land[land["session"] == sess]
        exp = pd.to_numeric(ls["experience_y"], errors="coerce").iloc[0] if len(ls) and "experience_y" in ls else np.nan
        sid = str(ls["subject_id"].dropna().iloc[0]) if len(ls) and ls["subject_id"].notna().any() else sess
        E = np.stack([rank_normal(g["land_y_kickside_m"].to_numpy(float)), rank_normal(g["land_z_m"].to_numpy(float))], -1)
        ks = {c: rank_normal(g[c].to_numpy(float)) for c in KNOBS if c in g}
        out.append(dict(session=sess, sid=sid, exp=float(exp), no=g["trial_no"].to_numpy(int), E=E, knobs=ks))
    return out


def _subject_stats(s: dict, n_perm: int, rng: np.random.Generator) -> dict[str, dict]:
    """被験者ごと・つまみごとの観測値と、シャッフルの帰無分布(つまみ間で同じシャッフルを使う)。"""
    n = len(s["E"])
    idx = np.argsort(rng.random((n_perm, n)), axis=1)
    Ep = s["E"][idx]  # (P, n, 2)
    res = {}
    for name, kv in s["knobs"].items():
        if np.isfinite(kv).sum() < MIN_PAIRS + 1:
            continue
        obs = path_stats(s["E"][None], kv[None])
        if not np.isfinite(obs["ind"]).any():
            continue
        res[name] = dict(obs={k: v[0] for k, v in obs.items()}, null=path_stats(Ep, kv[idx]))
    return res


def _pool(items: list[dict], what: str) -> tuple[np.ndarray, np.ndarray]:
    """被験者の平均(観測, 帰無)。1人しかいないつまみは返さない側で除く。"""
    obs = np.nanmean([it["obs"][what] for it in items], axis=0)
    null = np.nanmean([it["null"][what] for it in items], axis=0)
    return obs, null


def _p2(obs: np.ndarray, null: np.ndarray) -> np.ndarray:
    """両側 p(帰無分布の中心からの距離が観測以上の割合)。null は (P, ...)、obs は (...)。"""
    m = np.nanmean(null, axis=0)
    ok = np.isfinite(null)
    cnt = ((np.abs(null - m) >= np.abs(obs - m)) & ok).sum(0)
    return np.where(np.isfinite(obs), (1 + cnt) / (1 + ok.sum(0)), np.nan)


def analyze(trials: list[dict], n_perm: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    per = [(s, _subject_stats(s, n_perm, rng)) for s in trials]
    rows = []
    groups = [(s["session"], [(s, r)]) for s, r in per]
    groups += [("ALL", per), ("EXP", [(s, r) for s, r in per if np.isfinite(s["exp"]) and s["exp"] >= EXP_YEARS])]
    for gname, members in groups:
        if gname in ("ALL", "EXP") and len(members) < 2:
            continue
        knobs = [k for k in KNOBS if sum(k in r for _, r in members) >= (2 if gname in ("ALL", "EXP") else 1)]
        stat = {}
        for k in knobs:
            items = [r[k] for _, r in members if k in r]
            stat[k] = {w: _pool(items, w) for w in ("a", "b", "ind", "total")} | dict(n_subj=len(items))
        # max-T 補正: つまみ全体での |ind| の最大値(帰無の各シャッフルごと)
        maxnull = {}
        for ci, (c, c2) in enumerate(COMBOS):
            i, j = "yz".index(c), "yz".index(c2)
            mat = np.stack([np.abs(stat[k]["ind"][1][:, i, j] - np.nanmean(stat[k]["ind"][1][:, i, j])) for k in knobs], 1) if knobs else None
            maxnull[(i, j)] = np.nanmax(mat, axis=1) if mat is not None else None
        for k in knobs:
            st = stat[k]
            p_a = _p2(st["a"][0], st["a"][1])
            for c, c2 in COMBOS:
                i, j = "yz".index(c), "yz".index(c2)
                obs, null = st["ind"][0][i, j], st["ind"][1][:, i, j]
                dev = abs(obs - np.nanmean(null))
                p_adj = (1 + (maxnull[(i, j)] >= dev).sum()) / (1 + len(null)) if np.isfinite(dev) else np.nan
                members_k = [r[k]["obs"]["ind"][i, j] for _, r in members if k in r]
                rows.append(dict(group=gname, knob=k, err=c, next_err=c2, n_subj=st["n_subj"], a=float(st["a"][0][i]), p_a=float(p_a[i]), b=float(st["b"][0][j]),
                                 ind=float(obs), p_ind=float(_p2(obs, null)), p_ind_adj=p_adj, total=float(st["total"][0][i, j]),
                                 n_pos=int(np.sum(np.array(members_k) > 0)), n_neg=int(np.sum(np.array(members_k) < 0))))
    return pd.DataFrame(rows)


def _heat(ax, M: np.ndarray, P: np.ndarray, rows: list[str], cols: list[str], title: str, thr: float = 0.1) -> None:
    ax.imshow(M, cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.2f}" + ("*" if P[i, j] < thr else ""), ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    ax.set_title(title, fontsize=9)


def plot_paths(df: pd.DataFrame, out_a: Path, out_ind: Path) -> None:
    """a(前の誤差 → 次のつまみ)と間接効果(前の誤差 → つまみ → 次の誤差)の、つまみ × 組合せの表(ALL と EXP)。"""
    knobs = [k for k in KNOBS if (df["knob"] == k).any()]
    groups = [g for g in ("ALL", "EXP") if (df["group"] == g).any()]
    if not knobs or not groups:
        return
    fig, axs = plt.subplots(1, len(groups), figsize=(4.2 * len(groups) + 3.2, 0.32 * len(knobs) + 2.2), squeeze=False)
    for ax, g in zip(axs[0], groups):
        d = df[(df["group"] == g) & (df["err"].isin(["y", "z"])) & (df["next_err"] == df["err"])].set_index(["knob", "err"])
        M = np.array([[d["a"].get((k, c), np.nan) for c in "yz"] for k in knobs])
        P = np.array([[d["p_a"].get((k, c), np.nan) for c in "yz"] for k in knobs])
        _heat(ax, M, P, knobs, ["error y(n)", "error z(n)"], f"{g}: knob(n+1) response to previous error (* p<0.1)")
    fig.tight_layout()
    save_figure(fig, out_a, dpi=150)
    plt.close(fig)
    fig, axs = plt.subplots(1, len(groups), figsize=(5.4 * len(groups) + 3.0, 0.32 * len(knobs) + 2.2), squeeze=False)
    cols = [f"{c}(n)->{c2}(n+1)" for c, c2 in COMBOS]
    for ax, g in zip(axs[0], groups):
        d = df[df["group"] == g].set_index(["knob", "err", "next_err"])
        M = np.array([[d["ind"].get((k, c, c2), np.nan) for c, c2 in COMBOS] for k in knobs])
        P = np.array([[d["p_ind"].get((k, c, c2), np.nan) for c, c2 in COMBOS] for k in knobs])
        _heat(ax, M, P, knobs, cols, f"{g}: indirect effect via knob (* p<0.1)")
    fig.tight_layout()
    save_figure(fig, out_ind, dpi=150)
    plt.close(fig)


def run_mediation(sessions: list[str] | None = None, n_perm: int = 2000, seed: int = 0, root: Path | None = None, log=print) -> pd.DataFrame:
    root = root or ROOT / "output"
    trials = load_trials(sessions, root)
    if not trials:
        raise SystemExit(f"着弾のある被験者が {MIN_PAIRS + 1} 試行以上ありません")
    df = analyze(trials, n_perm=n_perm, seed=seed)
    tdir = root / "analysis" / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tdir / "mediation.csv", index=False, float_format="%.4f")
    for g in ("ALL", "EXP"):
        d = df[df["group"] == g]
        if d.empty:
            continue
        log(f"[{g}] 間接効果(前の誤差 → つまみ → 次の誤差)を p の小さい順に(p=シャッフル検定、adj=つまみ全体のmax-T補正。被験者が3人未満のつまみは除く)")
        top = d[d["n_subj"] >= min(3, d["n_subj"].max())].sort_values(["p_ind", "ind"], key=lambda s: s if s.name == "p_ind" else -s.abs()).head(8)
        for _, r in top.iterrows():
            log(f"   {r['err']}(n)→{r['knob']}→{r['next_err']}(n+1): a={r['a']:+.2f}(p={r['p_a']:.2f}) b={r['b']:+.2f} 間接={r['ind']:+.2f}"
                f"(p={r['p_ind']:.3f}, adj={r['p_ind_adj']:.2f}) 被験者{int(r['n_subj'])}人 符号 +{r['n_pos']}/−{r['n_neg']}")
    plot_paths(df, root / "analysis" / "landing" / "mediation_a.png", root / "analysis" / "landing" / "mediation_indirect.png")
    log(f"保存: {tdir / 'mediation.csv'} / {root / 'analysis' / 'landing'}(mediation_*.png)")
    return df
