"""試行間の依存: 前の試行の着弾のずれが、次の試行にどう効くか(誤差の補正)。  python run.py sequential

入力: config/landing.csv(試行番号順 = 実行順と仮定)。y は蹴り足側が +(左足は反転)、z は上が +。
被験者ごとに平均を引いた着弾 e(n) = (y, z) について、

1. ラグ1自己相関 ACF(1)(y・z・分布の長軸方向・短軸方向)。van Beers ら 2013(PLoS ONE)の指標:
   正 = 補正不足(前回に近い所へ行く)、0 = 最適、負 = 補正しすぎ(前回と反対側へ行くジグザグ)。
2. 補正係数行列 K(2×2): e(n+1) ≈ K e(n)。対角 = 同じ方向の補正、非対角 = クロス補正
   (K_yz = 上下のずれが次の左右に効く、K_zy = 左右のずれが次の上下に効く)。

試行数が少ないと ACF(1) は 0 より小さい側に偏る(約 -1/(n-1))ため、**被験者内で試行の順序をシャッフルした帰無分布**と比べる
(excess = 実測 - シャッフルの平均、p はシャッフルで実測以下/以上になる割合)。K の p も同じシャッフルで求める。
被験者をまとめた値(ALL、EXP = 経験8年以上)は、被験者ごとに独立にシャッフルして求める。

3. 傾向(ドリフト)を除いた版(`*_detrend`): 試行とともに着弾が一方向へ動く(疲労・学習・風など)と、補正がなくても ACF(1) は正に偏り、
   補正の負の ACF(1) を隠す。試行番号に対する直線を引いて残差で同じ解析をする(残差の順序をシャッフルした帰無分布と比べる)。
   傾向の大きさは `sequential_trend.csv`(傾き・Spearman)。分布の長軸/短軸は元の着弾の共分散から決める。
   帰無分布は、残差をそのままシャッフルするのでは**足りない**(直線を引いた残差は、独立な雑音でも ACF(1) が余計に負になる:
   n=10 で平均 -0.20、残差のシャッフルは -0.10)。そこで Freedman-Lane 法 = 「直線 + 残差のシャッフル」を作り直して、
   もう一度直線を引いた残差の ACF(1) を帰無分布とする。
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
from .kickfactors import load_landing
from scipy.stats import spearmanr
from .sessions import list_sessions, load_session

MIN_N = 5  # これ未満の試行数の被験者は対象外
EXP_YEARS = 8
KCOND_WARN = 10.0  # e(n) の二乗和行列の条件数(最大/最小固有値)がこれを超える(y と z の相関が約0.8以上)と、K の個々の成分は不安定
KEYS = ("yy", "yz", "zy", "zz")  # K[i, j]: 次の i 方向 ← 今の j 方向


def acf1(x: np.ndarray) -> float:
    """ラグ1自己相関(平均を引いて、分母は全体の二乗和)。"""
    x = np.asarray(x, float)
    x = x - x.mean()
    d = float((x * x).sum())
    return float((x[1:] * x[:-1]).sum() / d) if d > 0 else float("nan")


def _acf_rows(X: np.ndarray) -> np.ndarray:
    """行ごとの ACF(1)。X は (行, n) で、各行の平均は 0 に揃えてある前提。"""
    return (X[:, 1:] * X[:, :-1]).sum(1) / (X * X).sum(1)


def _suff(Y: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """行ごとの回帰の十分統計 Sxx = X'X, Sxt = X'T(X = e(n), T = e(n+1))。Y, Z は (行, n) の平均0の系列。"""
    X = np.stack([Y[:, :-1], Z[:, :-1]], -1)  # (行, n-1, 2)
    T = np.stack([Y[:, 1:], Z[:, 1:]], -1)
    return np.einsum("rni,rnj->rij", X, X), np.einsum("rni,rnj->rij", X, T)


def _K(Sxx: np.ndarray, Sxt: np.ndarray) -> np.ndarray:
    """(行, 2, 2) の K。e(n+1) = K e(n)。"""
    return np.swapaxes(np.linalg.solve(Sxx, Sxt), 1, 2)


def lag_matrix(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """平均を引いた (y, z) の系列から、e(n+1) ≈ K e(n) の K(2×2)を最小二乗で求める。"""
    Y = (np.asarray(y, float) - np.mean(y))[None]
    Z = (np.asarray(z, float) - np.mean(z))[None]
    Sxx, Sxt = _suff(Y, Z)
    return _K(Sxx, Sxt)[0]


def load_series(sessions: list[str] | None = None) -> list[dict]:
    """landing.csv から被験者ごとの系列(試行番号順、y は蹴り足側 +)。"""
    land = load_landing().dropna(subset=["y_m", "z_m"])
    out = []
    for sess in sessions or list(dict.fromkeys(land["session"])):
        g = land[land["session"] == sess].copy()
        g["no"] = g["trial"].astype(str).str.extract(r"(\d+)$")[0].astype(float)
        g = g.dropna(subset=["no"]).sort_values("no")
        if len(g) < MIN_N:
            continue
        side = str(g["kick_side"].dropna().iloc[0]).strip().lower() if "kick_side" in g and g["kick_side"].notna().any() else (
            load_session(sess).kick_side if sess in list_sessions() else "right")
        sid = str(g["subject_id"].dropna().iloc[0]) if "subject_id" in g and g["subject_id"].notna().any() else sess
        exp = pd.to_numeric(g["experience_y"], errors="coerce").iloc[0] if "experience_y" in g else np.nan
        out.append(dict(session=sess, sid=sid, exp=float(exp), no=g["no"].to_numpy(int),
                        y=(1.0 if side == "right" else -1.0) * g["y_m"].to_numpy(float), z=g["z_m"].to_numpy(float)))
    return out


def _perm_idx(n: int, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    return np.argsort(rng.random((n_perm, n)), axis=1)


def _axes(y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """分布の共分散の主軸(長軸, 短軸)の単位ベクトル。"""
    w, V = np.linalg.eigh(np.cov(np.vstack([y, z])))
    return V[:, np.argmax(w)], V[:, np.argmin(w)]


def _summ(obs: float, null: np.ndarray) -> dict:
    null = null[np.isfinite(null)]
    n = len(null)
    return dict(null_mean=float(null.mean()), null_lo=float(np.percentile(null, 2.5)), null_hi=float(np.percentile(null, 97.5)),
                excess=float(obs - null.mean()), p_neg=float((1 + (null <= obs).sum()) / (n + 1)),
                p_pos=float((1 + (null >= obs).sum()) / (n + 1)))


def detrend(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """試行番号 t に対する直線を引いた残差(平均は 0)。"""
    t = np.asarray(t, float)
    A = np.c_[np.ones_like(t), t - t.mean()]
    return np.asarray(x, float) - A @ np.linalg.lstsq(A, np.asarray(x, float), rcond=None)[0]


def trend_table(series: list[dict]) -> pd.DataFrame:
    """被験者ごとの、試行とともに着弾が動く傾向(1試行あたりの傾き m/試行、Spearman の ρ と p)。"""
    rows = []
    for s in series:
        row = dict(session=s["session"], subject_id=s["sid"], n=len(s["no"]))
        for ax in ("y", "z"):
            slope = np.polyfit(s["no"], s[ax], 1)[0]
            rho, p = spearmanr(s["no"], s[ax])
            row.update({f"slope_{ax}_m_per_trial": float(slope), f"rho_{ax}": float(rho), f"p_{ax}": float(p)})
        rows.append(row)
    return pd.DataFrame(rows)


def analyze(series: list[dict], n_perm: int = 10000, seed: int = 0, detrend_on: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """被験者ごと・グループ(ALL, EXP)の ACF(1) と K を、順序シャッフルの帰無分布つきで返す。detrend_on=True なら、試行番号に対する直線を除いた残差で行う。"""
    rng = np.random.default_rng(seed)
    acf_rows, k_rows = [], []
    per = []  # グループ集計用に被験者ごとの系列と帰無分布を持つ
    for s in series:
        y, z = s["y"] - s["y"].mean(), s["z"] - s["z"].mean()
        major, minor = _axes(y, z)  # 長軸/短軸は元の着弾の分布から(傾向を除く場合も同じ軸で見る)
        if detrend_on:
            y, z = detrend(s["y"], s["no"]), detrend(s["z"], s["no"])
        n = len(y)
        idx = _perm_idx(n, n_perm, rng)
        if detrend_on:  # Freedman-Lane: 直線 + シャッフルした残差 → 直線を引き直した残差(直線を引くことによる負の偏りも帰無分布に入る)
            t = np.asarray(s["no"], float)
            A = np.c_[np.ones_like(t), t - t.mean()]
            Mh = np.eye(n) - A @ np.linalg.pinv(A)
            Yp = ((s["y"] - y)[None] + y[idx]) @ Mh.T
            Zp = ((s["z"] - z)[None] + z[idx]) @ Mh.T
        else:
            Yp, Zp = y[idx], z[idx]
        proj = {"y": (y, Yp), "z": (z, Zp), "major": (major[0] * y + major[1] * z, major[0] * Yp + major[1] * Zp),
                "minor": (minor[0] * y + minor[1] * z, minor[0] * Yp + minor[1] * Zp)}
        null_acf = {}
        for ax, (x, xp) in proj.items():
            o = acf1(x)
            null_acf[ax] = _acf_rows(xp)
            acf_rows.append(dict(group=s["session"], subject_id=s["sid"], experience_y=s["exp"], n=n, axis=ax, acf1=o, **_summ(o, null_acf[ax])))
        Sxx, Sxt = _suff(y[None], z[None])
        Sxx_p, Sxt_p = _suff(Yp, Zp)
        per.append(dict(s=s, null_acf=null_acf, S=(Sxx, Sxt), Sp=(Sxx_p, Sxt_p), n=n))
        K, Kp = _K(Sxx, Sxt)[0], _K(Sxx_p, Sxt_p)
        for key, (i, j) in zip(KEYS, [(0, 0), (0, 1), (1, 0), (1, 1)]):
            k_rows.append(_k_row(s["session"], s["sid"], n - 1, key, K[i, j], Kp[:, i, j], np.linalg.cond(Sxx[0])))
    for gname, pick in (("ALL", lambda p: True), ("EXP", lambda p: np.isfinite(p["s"]["exp"]) and p["s"]["exp"] >= EXP_YEARS)):
        sub = [p for p in per if pick(p)]
        if len(sub) < 2:
            continue
        for ax in ("y", "z", "major", "minor"):
            o = float(np.mean([next(r["acf1"] for r in acf_rows if r["group"] == p["s"]["session"] and r["axis"] == ax) for p in sub]))
            null = np.mean([p["null_acf"][ax] for p in sub], axis=0)
            acf_rows.append(dict(group=gname, subject_id="+".join(p["s"]["sid"] for p in sub), experience_y=np.nan,
                                 n=sum(p["n"] for p in sub), axis=ax, acf1=o, **_summ(o, null)))
        Sxx = sum(p["S"][0] for p in sub)
        Sxt = sum(p["S"][1] for p in sub)
        Sxx_p = sum(p["Sp"][0] for p in sub)
        Sxt_p = sum(p["Sp"][1] for p in sub)
        K, Kp = _K(Sxx, Sxt)[0], _K(Sxx_p, Sxt_p)
        for key, (i, j) in zip(KEYS, [(0, 0), (0, 1), (1, 0), (1, 1)]):
            k_rows.append(_k_row(gname, "+".join(p["s"]["sid"] for p in sub), sum(p["n"] - 1 for p in sub), key, K[i, j], Kp[:, i, j], np.linalg.cond(Sxx[0])))
    return pd.DataFrame(acf_rows), pd.DataFrame(k_rows)


def _k_row(group: str, sid: str, n_pairs: int, key: str, val: float, null: np.ndarray, cond: float) -> dict:
    return dict(group=group, subject_id=sid, n_pairs=n_pairs, entry=key, value=float(val), cond=float(cond), null_mean=float(null.mean()),
                null_lo=float(np.percentile(null, 2.5)), null_hi=float(np.percentile(null, 97.5)),
                p_two=float((1 + (np.abs(null - null.mean()) >= abs(val - null.mean())).sum()) / (len(null) + 1)))


# ---------------------------------------------------------------- 図
COLS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6b6b66"]


def plot_series(series: list[dict], out: Path, trend: bool = False) -> None:
    """試行の順に y(蹴り足側 +)と z を並べる(ジグザグ = 補正の繰り返し)。"""
    n = len(series)
    fig, axs = plt.subplots(2, n, figsize=(3.1 * n, 5.2), sharey="row", squeeze=False)
    for c, s in enumerate(series):
        for r, (key, lab) in enumerate((("y", "Horizontal y, kicking-leg side + (m)"), ("z", "Vertical z (m)"))):
            ax = axs[r, c]
            v = s[key]
            ax.plot(s["no"], v, "-o", color=COLS[c % len(COLS)], ms=4, lw=1.2)
            if trend:
                ax.plot(s["no"], np.polyval(np.polyfit(s["no"], v, 1), s["no"]), color="#e0301e", ls="--", lw=1.2)
            else:
                ax.axhline(v.mean(), color="#6b6b66", ls="--", lw=0.8)
            ax.set_xlabel("Trial number")
            if c == 0:
                ax.set_ylabel(lab)
            ax.grid(alpha=0.25)
            if r == 0:
                ax.set_title(f"{s['sid']} ({s['session']}), n={len(v)}", fontsize=9)
    fig.tight_layout()
    save_figure(fig, out, dpi=150)
    plt.close(fig)


def plot_acf(acf: pd.DataFrame, out: Path, note: str = "") -> None:
    """ACF(1)の実測(点)とシャッフル帰無分布の95%範囲(棒)。範囲の外 = 順序に意味のある依存。"""
    groups = list(dict.fromkeys(acf["group"]))
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.0), sharey=True)
    for ax, (key, title) in zip(axs, (("y", "Horizontal y (kicking-leg side +)"), ("z", "Vertical z"),
                                      ("major", "Ellipse major axis"), ("minor", "Ellipse minor axis"))):
        for i, g in enumerate(groups):
            r = acf[(acf["group"] == g) & (acf["axis"] == key)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.plot([i, i], [r["null_lo"], r["null_hi"]], color="#b7b7b0", lw=7, solid_capstyle="butt")
            ax.plot(i, r["acf1"], "o", color="#e0301e" if (r["acf1"] < r["null_lo"] or r["acf1"] > r["null_hi"]) else "#1a1a19", ms=7, zorder=3)
        ax.axhline(0, color="#6b6b66", lw=0.8)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g if g in ("ALL", "EXP") else acf[acf["group"] == g]["subject_id"].iloc[0] for g in groups], fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, axis="y")
    axs[0].set_ylabel("Lag-1 autocorrelation")
    fig.suptitle("Dot = observed, gray bar = 95% range after shuffling trial order (red dot = outside)" + note, fontsize=10)
    fig.tight_layout()
    save_figure(fig, out, dpi=150)
    plt.close(fig)


def plot_k(kdf: pd.DataFrame, out: Path, note: str = "") -> None:
    """補正係数行列 K(行 = 次の試行、列 = 今の試行)。シャッフルの95%範囲の外の成分に * を付ける。"""
    groups = list(dict.fromkeys(kdf["group"]))
    fig, axs = plt.subplots(1, len(groups), figsize=(2.6 * len(groups), 3.3), squeeze=False)
    for ax, g in zip(axs[0], groups):
        d = kdf[kdf["group"] == g].set_index("entry")
        M = np.array([[d.loc["yy", "value"], d.loc["yz", "value"]], [d.loc["zy", "value"], d.loc["zz", "value"]]])
        ax.imshow(M, cmap="RdBu_r", vmin=-1.5, vmax=1.5)
        for i, a in enumerate("yz"):
            for j, b in enumerate("yz"):
                r = d.loc[a + b]
                star = "*" if (r["value"] < r["null_lo"] or r["value"] > r["null_hi"]) else ""
                ax.text(j, i, f"{M[i, j]:+.2f}{star}", ha="center", va="center", fontsize=11, color="#1a1a19")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["y(n)", "z(n)"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["y(n+1)", "z(n+1)"])
        ax.set_title((g if g in ("ALL", "EXP") else d["subject_id"].iloc[0]) + ("  (unstable)" if d["cond"].iloc[0] > KCOND_WARN else ""), fontsize=10)
    fig.suptitle("Correction matrix K: next error = K x current error (* = outside 95% range of shuffled order)" + note, fontsize=10)
    fig.tight_layout()
    save_figure(fig, out, dpi=150)
    plt.close(fig)


def run_sequential(sessions: list[str] | None = None, n_perm: int = 10000, seed: int = 0, root: Path | None = None,
                   log=print, detrend_on: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """detrend_on=True: 試行番号に対する直線(傾向)を除いた版。出力名に `_detrend` が付く。"""
    root = root or ROOT / "output"
    series = load_series(sessions)
    if not series:
        raise SystemExit(f"config/landing.csv に {MIN_N} 試行以上の被験者がいません")
    sfx = "_detrend" if detrend_on else ""
    acf, kdf = analyze(series, n_perm=n_perm, seed=seed, detrend_on=detrend_on)
    tdir, fdir = root / "analysis" / "tables", root / "analysis" / "landing"
    tdir.mkdir(parents=True, exist_ok=True)
    acf.to_csv(tdir / f"sequential_acf{sfx}.csv", index=False, float_format="%.4f")
    kdf.to_csv(tdir / f"sequential_K{sfx}.csv", index=False, float_format="%.4f")
    if detrend_on:
        tr = trend_table(series)
        tr.to_csv(tdir / "sequential_trend.csv", index=False, float_format="%.4f")
        for r in tr.itertuples():
            log(f"[{r.session}] 傾向(1試行あたり): y {r.slope_y_m_per_trial:+.2f} m (ρ={r.rho_y:+.2f}, p={r.p_y:.2f}) / z {r.slope_z_m_per_trial:+.2f} m (ρ={r.rho_z:+.2f}, p={r.p_z:.2f})")
    tag = "傾向を除く" if detrend_on else "元の着弾"
    for g in dict.fromkeys(acf["group"]):
        row = lambda ax: acf[(acf["group"] == g) & (acf["axis"] == ax)].iloc[0]  # noqa: E731
        ry, rz = row("y"), row("z")
        log(f"[{g}] ({tag}) n={int(ry['n'])} ACF(1) 左右y {ry['acf1']:+.2f} (シャッフル平均 {ry['null_mean']:+.2f}, 負側p={ry['p_neg']:.3f}) / "
            f"上下z {rz['acf1']:+.2f} ({rz['null_mean']:+.2f}, p={rz['p_neg']:.3f}) / 長軸 {row('major')['acf1']:+.2f} 短軸 {row('minor')['acf1']:+.2f}")
        d = kdf[kdf["group"] == g].set_index("entry")
        cond = float(d["cond"].iloc[0])
        log("      K: " + "  ".join(f"{k}={d.loc[k, 'value']:+.2f}(p={d.loc[k, 'p_two']:.2f})" for k in KEYS)
            + (f"  [条件数 {cond:.0f}: y と z が強く相関していて K は不安定。値でなく p とシャッフル範囲で見る]" if cond > KCOND_WARN else ""))
    note = "; linear trend removed" if detrend_on else ""
    plot_series(series, fdir / f"sequential_series{sfx}.png", trend=detrend_on)
    plot_acf(acf, fdir / f"sequential_acf{sfx}.png", note)
    plot_k(kdf, fdir / f"sequential_K{sfx}.png", note)
    log(f"保存: {tdir / f'sequential_acf{sfx}.csv'}, {tdir / f'sequential_K{sfx}.csv'} / {fdir}(sequential_*{sfx}.png)")
    return acf, kdf
