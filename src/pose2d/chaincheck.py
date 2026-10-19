"""運動連鎖のタイミング(大腿の角速度ピーク - 下腿のピーク)と着弾の上下 z の関係の頑健性チェック。  python run.py chain

`factors` で3人に共通して強かった xs_thigh_peak_to_shank_peak_ms(大腿ピークが下腿より早いほど高く飛ぶ)について、
1. 測り方への頑健性: 同じ「大腿が先に動く量」を別の定義(平滑化・検出窓を狭める・大腿の屈曲成分・相互相関・大腿の減速時刻)で測り直し、
   z との順位相関が定義によらず同じか見る。検出窓の端に張り付いた試行(ピークが取れていない)の数も出す。
2. 交絡: 下腿・大腿の角速度ピーク(速さ)、蹴り脚下腿の前方振り出し角、(あれば2Dの足先速度・ボール速度)の影響を除いた偏順位相関を出す。
被験者内で順位に直して合算する(被験者の癖・計測の違いを除く)。入力は Xsens(data/xsens)と着弾(config/landing.csv)だけで、2D運動学は不要。
出力: output/analysis/tables/chain_check.csv(定義別・偏相関)と output/analysis/landing/chain_timing_scatter.png
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, sosfiltfilt

from .config import ROOT
from .figio import save_figure
from .kickfactors import load_landing
from .sessions import list_sessions
from . import xsensfactors as xsf
from .xsensfactors import FS, _omega_world, _peak, load_trial, sensor_ids, xsens_dir, xsens_factors

EDGE = 30  # xsensfactors の大腿ピーク検出窓(下腿ピークの前 30 サンプル = 0.3 s)
LAGS = {
    "L0_current": "現行: 大腿|ω|最大(下腿ピークの前0.3s〜後0.06s)",
    "L1_lowpass15": "15Hz平滑化した|ω|でのピーク",
    "L2_narrow": "検出窓を前0.2s〜後0sに狭める",
    "L3_flexion": "大腿の屈曲成分(スイング軸まわり)のピーク",
    "L4_xcorr": "大腿と下腿の|ω|波形の相互相関による遅れ",
    "L5_braking": "大腿|ω|が最も急に減る時刻(大腿の減速)",
}


def _parab(y: np.ndarray, i: int) -> float:
    return _peak(y, i) if 1 <= i < len(y) - 1 else float(i)


def _smooth(x: np.ndarray, hz: float = 15.0) -> np.ndarray:
    sos = butter(2, hz / (FS / 2), output="sos")
    return sosfiltfilt(sos, x)


def _gyro_norm(d: Path, sid: str) -> np.ndarray | None:
    """センサーの生ジャイロ |Gyr|(rad/s)を100Hzに揃えて返す(欠落フレームは線形補間)。Gyr 列が無い計測は None。"""
    f = next(iter(sorted(d.glob(f"*{sid}.txt"))), None)
    if f is None:
        return None
    df = pd.read_csv(f, comment="/").dropna(axis=1, how="all")
    if not {"Gyr_X", "Gyr_Y", "Gyr_Z"} <= set(df.columns):
        return None
    pc = df["PacketCounter"].to_numpy(float)
    idx = np.unwrap(pc * 2 * np.pi / 65536) * 65536 / (2 * np.pi)
    idx = np.round(idx - idx[0])
    keep = np.r_[True, np.diff(idx) > 0]
    g, idx = df[["Gyr_X", "Gyr_Y", "Gyr_Z"]].to_numpy(float)[keep], idx[keep].astype(int)
    full = np.arange(idx[-1] + 1)
    return np.linalg.norm(np.column_stack([np.interp(full, idx, g[:, i]) for i in range(3)]), axis=1)


def gyro_lags(task_dir: Path, ids: dict[str, str], kick_side: str) -> dict:
    """姿勢の差分ではなく生ジャイロで、L0 と同じ窓・同じ方法(L6)と 15Hz 平滑化版(L7)の時間差 [ms] を求める。"""
    K = "R" if kick_side == "right" else "L"
    gt, gs = _gyro_norm(task_dir, ids[f"up_{K}"]), _gyro_norm(task_dir, ids[f"leg_{K}"])
    if gt is None or gs is None:
        return {"L6_gyro": np.nan, "L7_gyro_lp15": np.nan}
    n = min(len(gt), len(gs))
    gt, gs = gt[:n], gs[:n]
    out = {}
    for key, (a, b) in (("L6_gyro", (gt, gs)), ("L7_gyro_lp15", (_smooth(gt), _smooth(gs)))):
        ip = int(np.argmax(b))
        t0 = max(0, ip - EDGE)
        j = int(np.argmax(a[t0:ip + 6]))
        out[key] = (_parab(a, t0 + j) - _parab(b, ip)) / FS * 1000
    return out


def lag_definitions(rot: dict, kick_side: str) -> dict:
    """大腿と下腿の角速度ピークの時間差 [ms](<0 = 大腿が先)を、複数の定義で返す。"""
    K = "R" if kick_side == "right" else "L"
    om = {p: _omega_world(rot[p]) for p in (f"up_{K}", f"leg_{K}")}
    gt, gs = np.linalg.norm(om[f"up_{K}"], axis=1), np.linalg.norm(om[f"leg_{K}"], axis=1)
    n = len(gs)
    ip = int(np.argmax(gs))
    tp = _parab(gs, ip)
    out: dict = {}
    t0 = max(0, ip - EDGE)
    seg = gt[t0:ip + 6]
    j = int(np.argmax(seg))
    out["L0_current"] = (_parab(gt, t0 + j) - tp) / FS * 1000
    out["edge_hit"] = int(j <= 1)  # ピークが検出窓の前端 = 窓の中に山が無い
    gts, gss = _smooth(gt), _smooth(gs)
    ips = int(np.argmax(gss))
    t1 = max(0, ips - EDGE)
    out["L1_lowpass15"] = (_parab(gts, t1 + int(np.argmax(gts[t1:ips + 6]))) - _parab(gss, ips)) / FS * 1000
    t2 = max(0, ip - 20)
    out["L2_narrow"] = (_parab(gt, t2 + int(np.argmax(gt[t2:ip + 1]))) - tp) / FS * 1000
    # 大腿の屈曲成分: 下腿のスイング主軸 r まわりの大腿の角速度
    a, b = max(0, ip - 30), ip + 3
    w = om[f"leg_{K}"][a:b]
    wts = np.linalg.norm(w, axis=1)
    val, vec = np.linalg.eigh((w * wts[:, None]).T @ w)
    r = vec[:, np.argmax(val)]
    r = -r if (w * wts[:, None]).sum(axis=0) @ r < 0 else r
    fl = om[f"up_{K}"] @ r
    out["L3_flexion"] = (_parab(fl, t0 + int(np.argmax(fl[t0:ip + 6]))) - tp) / FS * 1000
    # 相互相関: 下腿波形を s サンプルずらして大腿波形に合わせたときの s(<0 = 大腿が先)
    lo, hi = max(0, ip - 45), min(n, ip + 10)
    x, y = gt[lo:hi] - gt[lo:hi].mean(), gs[lo:hi] - gs[lo:hi].mean()
    shifts = np.arange(-30, 1)
    cc = []
    for s in shifts:
        xs_, ys_ = (x[:s], y[-s:]) if s < 0 else (x, y)
        cc.append(float(np.dot(xs_, ys_) / (np.linalg.norm(xs_) * np.linalg.norm(ys_) + 1e-12)) if len(xs_) > 5 else -1.0)
    k = int(np.argmax(cc))
    out["L4_xcorr"] = (float(shifts[k]) + (_peak(np.array(cc), k) - k if 1 <= k < len(cc) - 1 else 0.0)) / FS * 1000
    # 大腿の減速時刻: 平滑化した大腿|ω|の最大の下り勾配の時刻(ピーク以降、下腿ピークまで)
    jt = t1 + int(np.argmax(gts[t1:ips + 6]))
    d = np.gradient(gts)
    seg_d = d[jt:ips + 6] if ips + 6 > jt else d[jt:jt + 5]
    out["L5_braking"] = (jt + int(np.argmin(seg_d)) - _parab(gss, ips)) / FS * 1000
    return out


def _rank_within(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """被験者内の順位 → 標準正規スコア(被験者の癖・計測差を除いて合算するため)。"""
    out = df[["session"]].copy()
    for c in cols:
        out[c] = df.groupby("session")[c].transform(lambda s: stats.norm.ppf((s.rank() - 0.5) / s.notna().sum()))
    return out


def partial_spearman(df: pd.DataFrame, x: str, y: str, controls: list[str]) -> tuple[float, float, int]:
    """偏順位相関(x と y から controls の線形の影響を順位上で除く)。(rho, p, n)"""
    d = df[["session", x, y, *controls]].dropna()
    if len(d) < 6 + len(controls):
        return np.nan, np.nan, len(d)
    rk = _rank_within(d, [x, y, *controls])
    X = np.column_stack([np.ones(len(rk)), *[rk[c] for c in controls]]) if controls else np.ones((len(rk), 1))
    rx = rk[x] - X @ np.linalg.lstsq(X, rk[x], rcond=None)[0]
    ry = rk[y] - X @ np.linalg.lstsq(X, rk[y], rcond=None)[0]
    r = float(np.corrcoef(rx, ry)[0, 1])
    n, k = len(rk), len(controls)
    t = r * np.sqrt((n - 2 - k) / max(1e-12, 1 - r * r))
    return r, float(2 * stats.t.sf(abs(t), n - 2 - k)), n


def build_table(sessions: list[str], log=print) -> pd.DataFrame:
    land = load_landing()
    land["trial_no"] = land["trial"].str.extract(r"(\d+)$")[0].astype(float)
    land["land_y_kickside_m"] = land["y_m"] * land["kick_side"].map({"right": 1.0, "left": -1.0})
    land = land.rename(columns={"z_m": "land_z_m"})
    ids = sensor_ids()
    rows, xfs = [], []
    for sess in sessions:
        side = land.loc[land["session"] == sess, "kick_side"].iloc[0] if (land["session"] == sess).any() else None
        if side is None:
            continue
        xf = xsens_factors(sess, side, log=lambda m: None)
        if xf.empty:
            log(f"[{sess}] Xsens の使える試行なし")
            continue
        xfs.append(xf)
        d = xsens_dir(sess, side)
        for r in xf.itertuples():
            tdir = d / f"task{r.trial_no}"
            try:
                rot, filled = load_trial(tdir / "task" if (tdir / "task").is_dir() else tdir, ids)
                lags = lag_definitions(rot, side)
                lags.update(gyro_lags(tdir / "task" if (tdir / "task").is_dir() else tdir, ids, side))
            except Exception as e:  # noqa: BLE001
                log(f"[スキップ] {sess}/task{r.trial_no}: {type(e).__name__}: {e}")
                continue
            rows.append(dict(session=sess, trial_no=int(r.trial_no), **lags))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    xs = pd.concat(xfs)
    keep = ["session", "trial_no", "xs_shank_gyro_peak_dps", "xs_thigh_gyro_peak_dps", "xs_kick_shank_fwd_deg", "xs_kick_thigh_fwd_deg"]
    df = df.merge(xs[keep], on=["session", "trial_no"], how="left")
    df = df.merge(land[["session", "trial_no", "land_z_m", "land_y_kickside_m"]], on=["session", "trial_no"], how="left")
    p2d = ROOT / "output" / "analysis" / "tables" / "kick_factors.csv"
    if p2d.exists():  # 2D運動学があれば、足先速度・ボール速度も交絡の候補に加える
        kf = pd.read_csv(p2d)
        cols = [c for c in ("toe_speed_mps", "toe_vel_mps", "ball_speed_mps") if c in kf]
        if cols:
            df = df.merge(kf[["session", "trial_no", *cols]].drop_duplicates(["session", "trial_no"]), on=["session", "trial_no"], how="left")
    return df


def plot_scatter(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = {"kurogi": "#2a78d6", "fujii": "#eb6834", "kataoka": "#1baf7a", "yamada": "#eda100"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    d = df.dropna(subset=["land_z_m", "L0_current"])
    for s, g in d.groupby("session"):
        axes[0].scatter(g["L0_current"], g["land_z_m"], color=cols.get(s, "#6b6b66"), label=f"{s} (n={len(g)})", s=24)
        if len(g) >= 4:
            b = np.polyfit(g["L0_current"], g["land_z_m"], 1)
            xs_ = np.array([g["L0_current"].min(), g["L0_current"].max()])
            axes[0].plot(xs_, b[0] * xs_ + b[1], color=cols.get(s, "#6b6b66"), lw=1)
    axes[0].set_xlabel("thigh peak - shank peak (ms; negative = thigh first)", fontsize=8)
    axes[0].set_ylabel("landing z (m)", fontsize=8)
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.25)
    ctr = [c for c in ("xs_shank_gyro_peak_dps", "xs_thigh_gyro_peak_dps") if c in d]
    z = _rank_within(d.dropna(subset=["L0_current", "land_z_m", *ctr]), ["L0_current", "land_z_m", *ctr])
    X = np.column_stack([np.ones(len(z)), *[z[c] for c in ctr]])
    rx = z["L0_current"] - X @ np.linalg.lstsq(X, z["L0_current"], rcond=None)[0]
    ry = z["land_z_m"] - X @ np.linalg.lstsq(X, z["land_z_m"], rcond=None)[0]
    for s in z["session"].unique():
        m = (z["session"] == s).to_numpy()
        axes[1].scatter(rx[m], ry[m], color=cols.get(s, "#6b6b66"), s=24, label=s)
    axes[1].set_xlabel("lag (rank-normal, speed peaks regressed out)", fontsize=8)
    axes[1].set_ylabel("landing z (rank-normal, speed peaks regressed out)", fontsize=8)
    axes[1].set_title("partial relation within subject", fontsize=8)
    axes[1].grid(alpha=0.25)
    for ax in axes:
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    save_figure(fig, path, dpi=140)
    plt.close(fig)


POSTURE = [("xs_support_thigh_lat_deg", "land_z_m", "支持脚の大腿の蹴り足側への傾き × 上下z"),
           ("xs_kick_shank_fwd_deg", "land_diag_m", "蹴り脚の下腿の前方振り出し角 × 斜め方向"),
           ("xs_kick_shank_fwd_deg", "land_y_kickside_m", "蹴り脚の下腿の前方振り出し角 × 蹴り足側y"),
           ("xs_support_thigh_fwd_deg", "land_y_kickside_m", "支持脚の大腿の前方への傾き × 蹴り足側y")]
OFFSETS_MS = (-40, -20, 0, 20, 40)  # 姿勢を測る時刻を、インパクト(下腿ピーク+20ms)からずらす


def _pooled(d: pd.DataFrame, x: str, y: str) -> tuple[float, float, int]:
    return partial_spearman(d, x, y, [])


def gyro_section(d: pd.DataFrame, log, rows: list) -> None:
    g = d.dropna(subset=["L6_gyro", "L7_gyro_lp15", "land_z_m"]) if "L6_gyro" in d else d.iloc[0:0]
    if g["session"].nunique() < 1 or len(g) < 6:
        log("\n[4] 生ジャイロ: Gyr 列のある試行が足りません")
        return
    log(f"\n[4] 生ジャイロ(姿勢の差分でなく Gyr の |ω|)で測り直した時間差と z の順位相関(Gyr のある被験者: {', '.join(sorted(g['session'].unique()))}, n={len(g)}):")
    for key in ("L0_current", "L6_gyro", "L7_gyro_lp15"):
        per = {s: gg[key].corr(gg["land_z_m"], method="spearman") for s, gg in g.groupby("session") if len(gg) >= 5}
        r, p, n = _pooled(g, key, "land_z_m")
        rows.append(dict(kind="gyro", name=key, control="", n=n, rho=r, p=p, **{f"rho_{s}": v for s, v in per.items()}))
        log(f"  {key:<14} " + ", ".join(f"{s} {v:+.2f}" for s, v in per.items()) + f" | 合算 {r:+.2f} (p={p:.4f}, n={n})")
    agree = {s: gg["L0_current"].corr(gg["L6_gyro"], method="spearman") for s, gg in g.groupby("session") if len(gg) >= 5}
    log("  L0(姿勢差分)とL6(生ジャイロ)の一致(試行間のSpearman): " + ", ".join(f"{s} {v:+.2f}" for s, v in agree.items()))


def posture_section(d: pd.DataFrame, log, rows: list) -> None:
    """支持脚の大腿の傾き・蹴り脚の下腿の振り出し角が着弾と関係する、という手がかりの頑健性: 姿勢を測る時刻をずらす・1人ずつ除く。"""
    land = load_landing()
    sides = land.drop_duplicates("session").set_index("session")["kick_side"].to_dict()
    base = d[["session", "trial_no", "land_z_m", "land_y_kickside_m"]].dropna().copy()
    base["land_diag_m"] = base.groupby("session").apply(lambda g: ((g["land_y_kickside_m"] - g["land_y_kickside_m"].mean()) + (g["land_z_m"] - g["land_z_m"].mean())) / np.sqrt(2)).reset_index(level=0, drop=True)
    frames = []
    old = xsf.SHANK_TO_IMPACT_S
    try:
        for off in OFFSETS_MS:
            xsf.SHANK_TO_IMPACT_S = old + off / 1000
            xs = pd.concat([xsens_factors(s, sides[s], log=lambda m: None) for s in base["session"].unique()])
            xs["offset_ms"] = off
            frames.append(xs)
    finally:
        xsf.SHANK_TO_IMPACT_S = old
    xs = pd.concat(frames).merge(base, on=["session", "trial_no"])
    log("\n[5] 姿勢の手がかりの頑健性(測る時刻を +-40ms ずらす / 1人ずつ除く)。rho は被験者内で順位化して合算、( )内は被験者別:")
    for fac, tgt, name in POSTURE:
        log(f"  {name}")
        for off in OFFSETS_MS:
            g = xs[xs["offset_ms"] == off]
            if len(g) < 10:
                continue
            r, p, n = _pooled(g, fac, tgt)
            per = {s: gg[fac].corr(gg[tgt], method="spearman") for s, gg in g.groupby("session") if len(gg) >= 5}
            loso = {s: _pooled(g[g["session"] != s], fac, tgt)[0] for s in g["session"].unique()}
            rows.append(dict(kind="posture", name=fac, control=f"{tgt}@{off:+d}ms", n=n, rho=r, p=p, **{f"rho_{s}": v for s, v in per.items()}))
            log(f"    {off:+4d}ms: 合算 {r:+.2f} (p={p:.3f}, n={n}) (" + ", ".join(f"{s} {v:+.2f}" for s, v in per.items())
                + ") | 除く→ " + ", ".join(f"-{s} {v:+.2f}" for s, v in loso.items()))


def run_chain(root: Path | None = None, log=print) -> pd.DataFrame:
    root = root or ROOT / "output"
    sessions = [s for s in list_sessions() if xsens_dir(s, "left") or xsens_dir(s, "right")]
    df = build_table(sessions, log)
    if df.empty or "land_z_m" not in df:
        log("Xsens と着弾がそろった試行がありません")
        return df
    (root / "analysis" / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(root / "analysis" / "tables" / "chain_check_trials.csv", index=False, float_format="%.3f")
    d = df.dropna(subset=["land_z_m"])
    log("\n[1] 時間差(L0 現行)の分布 [ms] と、検出窓の端に張り付いた試行(山が窓内に無い):")
    for s, g in d.groupby("session"):
        log(f"  {s}: n={len(g)} 平均 {g['L0_current'].mean():+.0f} SD {g['L0_current'].std():.0f} 範囲 {g['L0_current'].min():+.0f}〜{g['L0_current'].max():+.0f}  窓端 {int(g['edge_hit'].sum())} 試行")
    rows = []
    log("\n[2] 定義ごとの z との順位相関(被験者別 / 被験者内で順位化して合算):")
    for key, desc in LAGS.items():
        per = {s: g[key].corr(g["land_z_m"], method="spearman") for s, g in d.groupby("session") if len(g) >= 5}
        r, p, n = partial_spearman(d, key, "land_z_m", [])
        rows.append(dict(kind="definition", name=key, control="", n=n, rho=r, p=p, **{f"rho_{s}": v for s, v in per.items()}))
        log(f"  {key:<14} {desc}\n      " + ", ".join(f"{s} {v:+.2f}" for s, v in per.items()) + f" | 合算 {r:+.2f} (p={p:.4f}, n={n})")
    keys = list(LAGS)
    rk = _rank_within(d.dropna(subset=keys + ["land_z_m"]), keys + ["land_z_m"])
    cm = rk[keys].corr()
    log("\n    定義どうしの相関(被験者内で順位化して合算): " + " / ".join(f"{a[:2]}-{b[:2]} {cm.loc[a, b]:+.2f}" for a, b in [("L0_current", k) for k in keys[1:]]))
    # 6つの定義のうち最も強い |rho| が偶然に出る確率(着弾 z を被験者内でシャッフルする置換検定)
    rng = np.random.default_rng(0)
    obs = max(abs(float(np.corrcoef(rk[k], rk["land_z_m"])[0, 1])) for k in keys)
    cnt = 0
    for _ in range(5000):
        zp = rk.groupby("session")["land_z_m"].transform(lambda s: rng.permutation(s.to_numpy()))
        cnt += max(abs(float(np.corrcoef(rk[k], zp)[0, 1])) for k in keys) >= obs
    log(f"    6つの定義のうち最大の |rho|={obs:.2f} が偶然に出る確率(置換検定, 被験者内): p={(cnt + 1) / 5001:.4f}")
    rows.append(dict(kind="permutation", name="max_over_definitions", control="", n=len(rk), rho=obs, p=(cnt + 1) / 5001))
    sets = {
        "下腿ピーク": ["xs_shank_gyro_peak_dps"],
        "大腿ピーク": ["xs_thigh_gyro_peak_dps"],
        "下腿+大腿ピーク": ["xs_shank_gyro_peak_dps", "xs_thigh_gyro_peak_dps"],
        "下腿の前方角": ["xs_kick_shank_fwd_deg"],
        "上記すべて": ["xs_shank_gyro_peak_dps", "xs_thigh_gyro_peak_dps", "xs_kick_shank_fwd_deg"],
    }
    for extra in ("toe_speed_mps", "toe_vel_mps", "ball_speed_mps"):
        if extra in d and d[extra].notna().sum() >= 12:
            sets[f"+{extra}"] = ["xs_shank_gyro_peak_dps", "xs_thigh_gyro_peak_dps", extra]
    log("\n[3] 偏順位相関(L0 と z。次の量の影響を除く):")
    for name, ctr in sets.items():
        r, p, n = partial_spearman(d, "L0_current", "land_z_m", ctr)
        per = {}
        for s, g in d.groupby("session"):
            gg = g.copy()
            gg["session"] = s
            rr, pp, nn = partial_spearman(gg, "L0_current", "land_z_m", ctr)
            per[s] = rr
        rows.append(dict(kind="partial", name="L0_current", control=name, n=n, rho=r, p=p, **{f"rho_{s}": v for s, v in per.items()}))
        log(f"  {name:<16} 合算 {r:+.2f} (p={p:.4f}, n={n}) | 被験者別 " + ", ".join(f"{s} {v:+.2f}" for s, v in per.items()))
    gyro_section(d, log, rows)
    posture_section(d, log, rows)
    res = pd.DataFrame(rows)
    p = root / "analysis" / "tables" / "chain_check.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(p, index=False, float_format="%.4f")
    log(f"\n保存: {p}")
    try:
        plot_scatter(d, root / "analysis" / "landing" / "chain_timing_scatter.png")
    except OSError as e:
        log(f"[警告] 図を保存できません: {e}")
    return res
