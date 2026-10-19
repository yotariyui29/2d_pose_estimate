"""タスク精度スコア: 着弾点(ゴール平面)から「与えられたタスクへの精度」を定義し、過去データで試算する。  python run.py score

タスク = ゴール中心を目標に全力のインステップ(距離 D=11 m)。熟練度の候補としての従属変数。数値は連続値で扱う(成否・分割は使わない)。

座標・原点(重要): config/landing.csv の z は、元の着弾図(data/landing/ID○○_ball.png)の縦軸。ゴール枠は z=-0.22〜2.22 m で、
ゴール中心(目標 ×)は (y,z)=(0,1.0)。つまり z=0 は地面の 0.22 m 上(ボール半径 0.11 m の2倍)で、**ボール中心(地面の約0.11 m 上)ではない**。
誤差は常に「目標からの差」で定義する: e_y = y(蹴り足側 +)、e_z = z - 1.0。新規計測の `landing_z_m`(プロトコルでは「ボール中心を原点とした高さ」)も、
目標を (0, 1.0) のままにすると z が 0.11 m ずれた定誤差になるので、**地面からの高さで記録し、目標は 1.22 m(ゴール中心)とする**ことを勧める。

スコア(被験者ごと、試行 n=1..N の誤差ベクトル e_n=(e_y,e_z) [m]):
- 試行ごと: 半径誤差 r_n=|e_n|、角度誤差 θ_n=atan(r_n/D)、被験者の平均からの偏差 d_n=e_n-mean(e)(被験者内で3D量と回帰するとき)。
- CE(定誤差)=|mean(e)|(狙いの偏り)、VE(変動誤差)=sqrt(var(e_y)+var(e_z))(ddof=1。再現性)、TE(総合誤差)=sqrt(mean(r²))。
  TE² = CE² + VE²(N-1)/N。左右 y と上下 z は別々の CE_y・VE_y・CE_z・VE_z も出す。
- 角度への換算: x[m] → atan(x/D) [deg]。
- 副: 95%楕円の面積(傾きがあると面積は主軸比に敏感)。球速(全力指示なので別軸): 球速 × 半径誤差の被験者内 Spearman(速度-精度のトレードオフ)。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, spearmanr, t as tdist
from scipy.stats import f as fdist

from .config import ROOT
from .figio import save_figure
from .kickfactors import load_landing
from .landing import ellipse
from .sessions import list_sessions, load_session

D = 11.0  # ゴール面までの距離 [m]
TARGET_Z = 1.0  # landing.csv の枠でのゴール中心の高さ(地面 -0.22 + 1.22)
NOISE = {"uniform ±5cm": ("uniform", 0.05), "Gaussian SD 5cm": ("normal", 0.05), "Gaussian SD 10cm": ("normal", 0.10)}


def deg(x_m: float) -> float:
    return float(np.degrees(np.arctan(x_m / D)))


def scores(ey: np.ndarray, ez: np.ndarray) -> dict:
    """誤差ベクトル(m)から CE・VE・TE など。"""
    n = len(ey)
    cey, cez = float(ey.mean()), float(ez.mean())
    ce = float(np.hypot(cey, cez))
    vey, vez = float(ey.std(ddof=1)), float(ez.std(ddof=1))
    ve = float(np.hypot(vey, vez))
    te = float(np.sqrt(np.mean(ey ** 2 + ez ** 2)))
    return dict(n=n, CE_y=cey, CE_z=cez, CE=ce, VE_y=vey, VE_z=vez, VE=ve, TE=te)


def load_errors(sessions: list[str] | None = None) -> list[dict]:
    land = load_landing().dropna(subset=["y_m", "z_m"])
    out = []
    for sess in sessions or list(dict.fromkeys(land["session"])):
        g = land[land["session"] == sess].copy()
        g["no"] = g["trial"].astype(str).str.extract(r"(\d+)$")[0].astype(float)
        g = g.sort_values("no")
        if len(g) < 5:
            continue
        side = str(g["kick_side"].dropna().iloc[0]).lower() if "kick_side" in g and g["kick_side"].notna().any() else (
            load_session(sess).kick_side if sess in list_sessions() else "right")
        exp = pd.to_numeric(g["experience_y"], errors="coerce").iloc[0] if "experience_y" in g else np.nan
        sid = str(g["subject_id"].dropna().iloc[0]) if "subject_id" in g and g["subject_id"].notna().any() else sess
        out.append(dict(session=sess, sid=sid, exp=float(exp), no=g["no"].to_numpy(int),
                        ey=(1.0 if side == "right" else -1.0) * g["y_m"].to_numpy(float), ez=g["z_m"].to_numpy(float) - TARGET_Z))
    return out


def _noise(shape, kind: str, scale: float, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-scale, scale, shape) if kind == "uniform" else rng.normal(0, scale, shape)


def scale_interval(obs: float, draws: np.ndarray) -> tuple[float, float]:
    """尺度(VE・TE)の95%区間。標本から標本を引き直した値 s* の分布は、母集団の値 σ ではなく標本の値 s のまわりに散らばり、小さい標本では s* が
    下側へ偏る(単純なパーセンタイル区間は幅が狭く、上側が足りない)。比 s*/s が s/σ に近い分布をもつとして区間を反転する: [s²/q97.5, s²/q2.5]。"""
    q_lo, q_hi = np.percentile(draws, [2.5, 97.5])
    return float(obs ** 2 / q_hi), float(obs ** 2 / q_lo)


def uncertainty(subjects: list[dict], n_boot: int = 4000, seed: int = 0) -> pd.DataFrame:
    """スコアの不確かさ: (a)測定誤差だけ(着弾の読み取り誤差を乱数で)、(b)試行の標本誤差だけ、(c)両方。95%区間と偏り。
    (b)(c) は、被験者の平均と共分散から正規分布で n 本を引き直して(パラメトリック・ブートストラップ。n=7〜11 で試行の引き直しは退化しやすい)、
    VE・TE の区間は `scale_interval` で反転する。CE はパーセンタイル。"""
    rng = np.random.default_rng(seed)
    rows = []
    for s in subjects:
        n = len(s["ey"])
        base = scores(s["ey"], s["ez"])
        for key in ("CE", "VE", "TE"):
            rows.append(dict(session=s["session"], subject_id=s["sid"], score=key, source="observed", scenario="", value=base[key], bias=0.0, lo=np.nan, hi=np.nan))
        for scen, (kind, scale) in {"none": ("normal", 0.0), **NOISE}.items():
            keys = ("CE", "VE", "TE")
            # (a) 測定誤差のみ
            if scen != "none":
                vals = {k: [] for k in keys}
                for _ in range(n_boot):
                    sc = scores(s["ey"] + _noise(n, kind, scale, rng), s["ez"] + _noise(n, kind, scale, rng))
                    for k in vals:
                        vals[k].append(sc[k])
                for k, v in vals.items():
                    v = np.array(v)
                    rows.append(dict(session=s["session"], subject_id=s["sid"], score=k, source="measurement noise only", scenario=scen, value=base[k],
                                     bias=float(v.mean() - base[k]), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5))))
            # (b)(c) ブートストラップ(+測定誤差)
            vals = {k: [] for k in keys}
            mu, cov = np.array([s["ey"].mean(), s["ez"].mean()]), np.cov(np.vstack([s["ey"], s["ez"]]))
            for _ in range(n_boot):
                ey, ez = rng.multivariate_normal(mu, cov, n).T
                if scen != "none":
                    ey, ez = ey + _noise(n, kind, scale, rng), ez + _noise(n, kind, scale, rng)
                sc = scores(ey, ez)
                for k in vals:
                    vals[k].append(sc[k])
            for k, v in vals.items():
                v = np.array(v)
                lo, hi = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if k == "CE" else scale_interval(base[k], v)
                rows.append(dict(session=s["session"], subject_id=s["sid"], score=k, source="sampling" if scen == "none" else "sampling + noise",
                                 scenario=scen if scen != "none" else "", value=base[k], bias=float(v.mean() - base[k]), lo=lo, hi=hi))
    return pd.DataFrame(rows)


def ve_power(ns=(10, 20, 30, 40, 50), ratios=(1.25, 1.5, 2.0)) -> pd.DataFrame:
    """2人の VE の比を、同じ本数 n で検出できる確率(F検定の概算、両側 α=0.05)。VE² の自由度は y,z が独立で等分散なら 2(n-1)、
    楕円が細長い(固有値比が大きい)と約 (n-1) まで下がるので両方を示す。"""
    rows = []
    for n in ns:
        for label, df in (("isotropic (df=2(n-1))", 2 * (n - 1)), ("elongated (df=n-1)", n - 1)):
            crit = fdist.ppf(0.975, df, df)
            for r in ratios:
                rho = r ** 2  # VE² の比
                power = float(1 - fdist.cdf(crit / rho, df, df) + fdist.cdf(1 / (crit * rho), df, df))
                rows.append(dict(n_per_subject=n, shape=label, ve_ratio=r, power=power))
    return pd.DataFrame(rows)


def noise_effect(ve_true: float, sigma: float) -> float:
    """測定誤差 σ(各軸)で VE がどれだけ過大に出るか(比)。VE_obs² = VE² + 2σ²。"""
    return float(np.sqrt(ve_true ** 2 + 2 * sigma ** 2) / ve_true)


def requirements(sens: pd.DataFrame, budgets=(0.1, 0.2, 0.3)) -> pd.DataFrame:
    """接触モデルの感度(contact_sensitivity.csv)から、「着弾の予測誤差 ≤ budget[m]」にするための入力の測定精度。
    4つの入力(足先速度の方位・仰角、接触点の左右・上下)が等しく寄与し、独立とする: σ_i = budget / (2 S_i)。"""
    s = sens.set_index("input")
    spec = [("az", "foot velocity azimuth (deg)", "dy_k_m_per_step", 1.0), ("elev", "foot velocity elevation (deg)", "dz_m_per_step", 1.0),
            ("h_lat", "contact offset, lateral (mm)", "dy_k_m_per_step", 0.01), ("h_up", "contact offset, vertical (mm)", "dz_m_per_step", 0.01)]
    rows = []
    for key, label, col, step in spec:
        sens_m_per_unit = abs(float(s.loc[key, col])) / step  # m / (deg or m)
        for b in budgets:
            sig = b / (2 * sens_m_per_unit)
            rows.append(dict(input=label, budget_m=b, required_sd=sig * (1000 if "mm" in label else 1), landing_per_unit_m=sens_m_per_unit * (0.001 if "mm" in label else 1)))
    return pd.DataFrame(rows)


def tracking_requirement(sigma_deg: float = 0.5, speed: float = 18.0, dts_ms=(3, 5, 10)) -> pd.DataFrame:
    """足先の方向を σ_deg で測るのに必要な位置の精度。方向は2点の差から求めるので σ_pos = tan(σ_deg) · L / √2、L = 速度 × 時間間隔。"""
    rows = []
    for dt in dts_ms:
        L = speed * dt / 1000
        rows.append(dict(speed_mps=speed, window_ms=dt, travel_mm=L * 1000, required_position_sd_mm=float(np.tan(np.radians(sigma_deg)) * L * 1000 / np.sqrt(2)), angle_sd_deg=sigma_deg))
    return pd.DataFrame(rows)


def speed_accuracy(subjects: list[dict], root: Path) -> dict[str, tuple[float, float, int]]:
    p = root / "analysis" / "tables" / "kick_factors.csv"
    if not p.exists():
        return {}
    kf = pd.read_csv(p)
    out = {}
    for s in subjects:
        g = kf[kf["session"] == s["session"]].set_index("trial_no")
        sel = [(g.loc[no, "ball_speed_mps"], r) for no, r in zip(s["no"], np.hypot(s["ey"], s["ez"])) if no in g.index and np.isfinite(g.loc[no, "ball_speed_mps"])]
        if len(sel) >= 5:
            rho, p_ = spearmanr([a for a, _ in sel], [b for _, b in sel])
            out[s["session"]] = (float(rho), float(p_), len(sel))
    return out


# ---------------------------------------------------------------- 軸ごと(左右 y・上下 z)
AXES = (("y", "Horizontal y (left-right)"), ("z", "Vertical z (up-down)"))


def axis_scores(e: np.ndarray) -> dict:
    """1軸のスコア: CE(符号つきの平均のずれ)、VE(SD、ddof=1)、TE(RMSE)。"""
    return dict(CE=float(e.mean()), VE=float(e.std(ddof=1)), TE=float(np.sqrt(np.mean(e ** 2))))


def axis_table(subjects: list[dict], n_boot: int = 4000, seed: int = 0) -> pd.DataFrame:
    """被験者×軸のスコアと95%区間(CE: t、VE: カイ二乗、TE: 引き直し)、読み取り誤差(SD 5・10cm)による VE の過大(比)。"""
    rng = np.random.default_rng(seed)
    rows = []
    for s in subjects:
        n = len(s["ey"])
        for ax, _ in AXES:
            e = s["e" + ax]
            sc = axis_scores(e)
            sims = np.array([axis_scores(rng.normal(e.mean(), e.std(ddof=1), n))["TE"] for _ in range(n_boot)])
            row = dict(session=s["session"], subject_id=s["sid"], experience_y=s["exp"], axis=ax, n=n,
                       **{k: sc[k] for k in ("CE", "VE", "TE")}, **{f"{k}_deg": deg(sc[k]) for k in ("CE", "VE", "TE")})
            # CE: t 区間、VE: カイ二乗区間(正規分布の仮定)、TE: 正規分布での引き直し + 尺度の区間
            h = float(tdist.ppf(0.975, n - 1) * sc["VE"] / np.sqrt(n))
            row["CE_lo"], row["CE_hi"] = sc["CE"] - h, sc["CE"] + h
            row["VE_lo"], row["VE_hi"] = float(sc["VE"] * np.sqrt((n - 1) / chi2.ppf(0.975, n - 1))), float(sc["VE"] * np.sqrt((n - 1) / chi2.ppf(0.025, n - 1)))
            row["TE_lo"], row["TE_hi"] = scale_interval(sc["TE"], sims)
            row["VE_robust"] = float(1.4826 * np.median(np.abs(e - np.median(e))))  # 外れ値に強い目安(MAD)。SD と大きく違うときは分布が偏っている
            for sig in (0.05, 0.10):
                row[f"VE_inflation_sd{int(sig * 100)}cm"] = float(np.sqrt(sc["VE"] ** 2 + sig ** 2) / sc["VE"])
            rows.append(row)
    return pd.DataFrame(rows)


def zorigin_sensitivity(subjects: list[dict], alt_target_z: float = 1.11) -> pd.DataFrame:
    """目標の高さの仮定(図の枠 1.0 m か、ボール中心が地面の 0.11 m 上なら 1.11 m か)で、上下方向の CE・TE がどう変わるか。VE は変わらない。"""
    rows = []
    for s in subjects:
        for tz in (TARGET_Z, alt_target_z):
            e = s["ez"] - (tz - TARGET_Z)
            sc = axis_scores(e)
            rows.append(dict(session=s["session"], subject_id=s["sid"], target_z_m=tz, CE_z=sc["CE"], VE_z=sc["VE"], TE_z=sc["TE"]))
    return pd.DataFrame(rows)


def requirements_by_axis(sens: pd.DataFrame, ve_y: float, ve_z: float, frac: float = 0.25) -> pd.DataFrame:
    """軸ごとの必要な測定精度。着弾の予測誤差(入力の測定誤差による)を、その軸の典型的な VE の frac 倍(分散で frac² 倍)以下にする。
    左右 y は方位・接触点の左右の2入力、上下 z は仰角・接触点の上下・足先速度の3入力が等しく独立に寄与するとして σ_i = 予算 / (√k S_i)。"""
    s = sens.set_index("input")
    spec = {"y": [("foot velocity azimuth (deg)", "az", "dy_k_m_per_step", 1.0, 1.0), ("contact offset, lateral (mm)", "h_lat", "dy_k_m_per_step", 0.01, 1000.0)],
            "z": [("foot velocity elevation (deg)", "elev", "dz_m_per_step", 1.0, 1.0), ("contact offset, vertical (mm)", "h_up", "dz_m_per_step", 0.01, 1000.0),
                  ("foot speed (m/s)", "speed", "dz_m_per_step", 1.0, 1.0)]}
    rows = []
    for ax, ve in (("y", ve_y), ("z", ve_z)):
        budget = frac * ve
        for label, key, col, step, mult in spec[ax]:
            S = abs(float(s.loc[key, col])) / step  # 着弾[m] / 入力の単位(deg, m, m/s)
            rows.append(dict(axis=ax, input=label, budget_m=budget, typical_ve_m=ve, landing_per_unit_m=S * (0.001 if mult == 1000.0 else 1.0),
                             required_sd=budget / (np.sqrt(len(spec[ax])) * S) * mult))
    return pd.DataFrame(rows)


def plot_axes(ax_tbl: pd.DataFrame, trials: pd.DataFrame, out: Path) -> None:
    """軸ごとの図: 左 = CE(符号つき)・VE・TE と95%区間、右 = 試行ごとのずれ(平均と ±VE の帯)。"""
    order = ax_tbl.drop_duplicates("session").sort_values("experience_y", ascending=False, na_position="last")["session"].tolist()
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.2))
    for r, (a, title) in enumerate(AXES):
        d = ax_tbl[ax_tbl["axis"] == a].set_index("session")
        ax = axs[r, 0]
        w = 0.26
        for k, (col, lab, c) in enumerate((("CE", "CE: constant error (signed)", "#2a78d6"), ("VE", "VE: variable error (SD)", "#eb6834"), ("TE", "TE: total error (RMSE)", "#1a1a19"))):
            vals = np.array([d.loc[s, col] for s in order])
            lo = np.array([d.loc[s, col + "_lo"] for s in order])
            hi = np.array([d.loc[s, col + "_hi"] for s in order])
            xx = np.arange(len(order)) + (k - 1) * w
            ax.bar(xx, vals, w, color=c, label=lab)
            ax.errorbar(xx, vals, yerr=[np.maximum(vals - lo, 0), np.maximum(hi - vals, 0)], fmt="none", ecolor="#6b6b66", lw=1.1, capsize=2)
        ax.axhline(0, color="#6b6b66", lw=0.8)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{d.loc[s, 'subject_id']}\n{'%g y' % d.loc[s, 'experience_y'] if np.isfinite(d.loc[s, 'experience_y']) else 'exp. ?'}" for s in order], fontsize=9)
        ax.set_ylabel("Error at the goal plane (m)")
        ax.set_title(f"{title}: scores (bars) with 95% interval", fontsize=10)
        if r == 0:
            ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.25, axis="y")
        ax = axs[r, 1]
        for i, s in enumerate(order):
            t = trials[trials["session"] == s]
            e = t["e_" + a].to_numpy()
            ax.axhspan(e.mean() - e.std(ddof=1), e.mean() + e.std(ddof=1), xmin=(i + 0.1) / len(order), xmax=(i + 0.9) / len(order), color="#e1e1dc", alpha=0.9)
            ax.scatter(np.full(len(e), i) + np.linspace(-0.15, 0.15, len(e)), e, s=24, color="#1a1a19", alpha=0.85, zorder=3)
            ax.plot([i - 0.25, i + 0.25], [e.mean()] * 2, color="#e0301e", lw=2, zorder=4)
        ax.axhline(0, color="#1a1a19", lw=1)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([ax_tbl[(ax_tbl["session"] == s)]["subject_id"].iloc[0] for s in order], fontsize=9)
        ax.set_ylabel("Error of each trial (m)")
        ax.set_title(f"{title}: trials (red = mean, gray = mean ± VE)", fontsize=10)
        ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    save_figure(fig, out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 図
def plot_scores(tbl: pd.DataFrame, trials: pd.DataFrame, out: Path) -> None:
    order = tbl.sort_values("experience_y", ascending=False, na_position="last")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axs[0]
    w = 0.26
    x = np.arange(len(order))
    for k, (col, lab, c) in enumerate((("CE", "CE: constant error (bias)", "#2a78d6"), ("VE", "VE: variable error (spread)", "#eb6834"), ("TE", "TE: total error", "#1a1a19"))):
        ax.bar(x + (k - 1) * w, order[col], w, color=c, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.subject_id}\n{'%g y' % r.experience_y if np.isfinite(r.experience_y) else 'exp. ?'}" for r in order.itertuples()], fontsize=9)
    ax.set_ylabel("Error at the goal plane (m)")
    sec = ax.secondary_yaxis("right", functions=(lambda m: np.degrees(np.arctan(m / D)), lambda d: D * np.tan(np.radians(d))))
    sec.set_ylabel("Angle seen from the ball (deg)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, axis="y")
    ax.set_title("Accuracy score per subject (11 m)", fontsize=10)
    ax = axs[1]
    for i, r in enumerate(order.itertuples()):
        t = trials[trials["session"] == r.session]
        ax.scatter(np.full(len(t), i) + np.linspace(-0.12, 0.12, len(t)), t["r"], s=22, color="#6b6b66", alpha=0.8)
        ax.plot([i - 0.2, i + 0.2], [t["r"].median()] * 2, color="#e0301e", lw=2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order["subject_id"], fontsize=9)
    ax.set_ylabel("Radial error per trial (m)")
    ax.set_title("Trial-level score (red = median)", fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    ax = axs[2]
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    for i, r in enumerate(order.itertuples()):
        t = trials[trials["session"] == r.session]
        ax.scatter(t["e_y"], t["e_z"], s=22, color=cols[i % len(cols)], alpha=0.85, label=r.subject_id)
        ax.add_patch(plt.Circle((t["e_y"].mean(), t["e_z"].mean()), r.VE, fill=False, ec=cols[i % len(cols)], lw=1.4))
    ax.plot(0, 0, "x", color="#e0301e", ms=9, mew=2.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Horizontal error, kicking-leg side + (m)  [left-footed subjects mirrored]")
    ax.set_ylabel("Vertical error (m)")
    ax.set_title("Errors from the target (circle: radius = VE)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, out, dpi=150)
    plt.close(fig)


def plot_uncertainty(unc: pd.DataFrame, out: Path) -> None:
    """VE と TE の95%区間: 標本(試行数)だけ・測定誤差だけ・両方。測定誤差が支配的でないことを示す。"""
    subs = list(dict.fromkeys(unc["subject_id"]))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=False)
    srcs = [("sampling", "", "#1a1a19", "trials only (resampling)"), ("measurement noise only", "Gaussian SD 10cm", "#eb6834", "reading error only (SD 10 cm)"),
            ("sampling + noise", "Gaussian SD 10cm", "#2a78d6", "trials + reading error (SD 10 cm)")]
    for ax, sc in zip(axs, ("VE", "TE")):
        for i, sid in enumerate(subs):
            obs = unc[(unc["subject_id"] == sid) & (unc["score"] == sc) & (unc["source"] == "observed")]["value"].iloc[0]
            for k, (src, scen, col, lab) in enumerate(srcs):
                r = unc[(unc["subject_id"] == sid) & (unc["score"] == sc) & (unc["source"] == src) & (unc["scenario"] == scen)]
                if r.empty:
                    continue
                r = r.iloc[0]
                x = i + (k - 1) * 0.2
                ax.plot([x, x], [r["lo"], r["hi"]], color=col, lw=5, solid_capstyle="butt", label=lab if i == 0 else None)
            ax.plot(i, obs, "o", color="white", mec="#1a1a19", ms=7, zorder=3)
        ax.set_xticks(range(len(subs)))
        ax.set_xticklabels(subs)
        ax.set_ylabel(f"{sc} (m)")
        ax.set_title(f"{sc}: 95% interval by source of uncertainty", fontsize=10)
        ax.grid(alpha=0.25, axis="y")
    axs[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    save_figure(fig, out, dpi=150)
    plt.close(fig)


def plot_requirements(req: pd.DataFrame, power: pd.DataFrame, out: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axs[0]
    inputs = list(dict.fromkeys(req["input"]))
    w = 0.26
    for k, b in enumerate(sorted(req["budget_m"].unique())):
        r = req[req["budget_m"] == b]
        ax.bar(np.arange(len(inputs)) + (k - 1) * w, r["required_sd"], w, label=f"landing prediction error at most {b:g} m")
    ax.set_xticks(range(len(inputs)))
    ax.set_xticklabels([i.replace(" (", "\n(") for i in inputs], fontsize=8)
    ax.set_ylabel("Required measurement SD (deg or mm)")
    ax.set_title("How precisely must each input be measured?", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    ax = axs[1]
    for (shape, ratio), g in power.groupby(["shape", "ve_ratio"]):
        ax.plot(g["n_per_subject"], g["power"], "-o" if shape.startswith("iso") else "--s", ms=4, label=f"VE ratio {ratio:g}, {shape.split(' ')[0]}")
    ax.axhline(0.8, color="#6b6b66", lw=0.8)
    ax.set_xlabel("Trials per subject")
    ax.set_ylabel("Power to detect a VE difference between two subjects")
    ax.set_title("How many kicks to compare two subjects' VE?", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, out, dpi=150)
    plt.close(fig)


def run_score(sessions: list[str] | None = None, n_boot: int = 4000, seed: int = 0, root: Path | None = None, log=print) -> pd.DataFrame:
    root = root or ROOT / "output"
    subjects = load_errors(sessions)
    if not subjects:
        raise SystemExit("config/landing.csv に5試行以上の被験者がいません")
    sp = speed_accuracy(subjects, root)
    rows, trows = [], []
    for s in subjects:
        sc = scores(s["ey"], s["ez"])
        el = ellipse(s["ey"], s["ez"]) if len(s["ey"]) >= 3 else {}
        row = dict(session=s["session"], subject_id=s["sid"], experience_y=s["exp"], **sc, A95_m2=np.pi * el.get("a", np.nan) * el.get("b", np.nan),
                   CE_deg=deg(sc["CE"]), VE_deg=deg(sc["VE"]), TE_deg=deg(sc["TE"]))
        if s["session"] in sp:
            row.update(speed_r_spearman=sp[s["session"]][0], speed_r_p=sp[s["session"]][1])
        rows.append(row)
        r = np.hypot(s["ey"], s["ez"])
        for i in range(len(r)):
            trows.append(dict(session=s["session"], subject_id=s["sid"], trial_no=int(s["no"][i]), e_y=s["ey"][i], e_z=s["ez"][i], r=r[i], theta_deg=deg(r[i]),
                              d_y=s["ey"][i] - s["ey"].mean(), d_z=s["ez"][i] - s["ez"].mean()))
    tbl, trials = pd.DataFrame(rows), pd.DataFrame(trows)
    unc = uncertainty(subjects, n_boot=n_boot, seed=seed)
    sens_p = root / "analysis" / "tables" / "contact_sensitivity.csv"
    req = requirements(pd.read_csv(sens_p)) if sens_p.exists() else pd.DataFrame()
    power = ve_power()
    trk = tracking_requirement()
    tdir, fdir = root / "analysis" / "tables", root / "analysis" / "landing"
    tdir.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(tdir / "accuracy_score.csv", index=False, float_format="%.4f")
    trials.to_csv(tdir / "accuracy_score_trials.csv", index=False, float_format="%.4f")
    unc.to_csv(tdir / "accuracy_score_uncertainty.csv", index=False, float_format="%.4f")
    power.to_csv(tdir / "accuracy_requirements_power.csv", index=False, float_format="%.3f")
    trk.to_csv(tdir / "accuracy_requirements_tracking.csv", index=False, float_format="%.3f")
    if not req.empty:
        req.to_csv(tdir / "accuracy_requirements_precision.csv", index=False, float_format="%.3f")
    for r in tbl.itertuples():
        log(f"[{r.session}] n={r.n} CE {r.CE:.2f} m ({r.CE_deg:.2f}°) / VE {r.VE:.2f} m ({r.VE_deg:.2f}°) / TE {r.TE:.2f} m ({r.TE_deg:.2f}°)  "
            f"[CE y{r.CE_y:+.2f} z{r.CE_z:+.2f}, VE y{r.VE_y:.2f} z{r.VE_z:.2f}]")
    axt = axis_table(subjects, n_boot=n_boot, seed=seed)
    axt.to_csv(tdir / "accuracy_score_axes.csv", index=False, float_format="%.4f")
    zs = zorigin_sensitivity(subjects)
    zs.to_csv(tdir / "accuracy_zorigin_sensitivity.csv", index=False, float_format="%.4f")
    if not req.empty:
        rq = requirements_by_axis(pd.read_csv(sens_p), float(tbl["VE_y"].median()), float(tbl["VE_z"].median()))
        rq.to_csv(tdir / "accuracy_requirements_axes.csv", index=False, float_format="%.3f")
    for a, _ in AXES:
        for r in axt[axt["axis"] == a].itertuples():
            log(f"  [{r.session}] {a}: CE {r.CE:+.2f} [{r.CE_lo:+.2f}, {r.CE_hi:+.2f}] / VE {r.VE:.2f} [{r.VE_lo:.2f}, {r.VE_hi:.2f}] / TE {r.TE:.2f} m ({r.TE_deg:.2f}°)")
    plot_axes(axt, trials, fdir / "accuracy_score_axes.png")
    plot_scores(tbl, trials, fdir / "accuracy_score.png")
    plot_uncertainty(unc, fdir / "accuracy_score_uncertainty.png")
    if not req.empty:
        plot_requirements(req, power, fdir / "accuracy_requirements.png")
    log(f"保存: {tdir}(accuracy_*.csv) / {fdir}(accuracy_*.png)")
    return tbl
