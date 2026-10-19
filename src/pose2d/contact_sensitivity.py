"""足-ボール接触モデルの感度解析。  python run.py contact

問い: 着弾(ゴール平面 y,z)のばらつきと斜めの傾きは、足の速度方向のばらつき(運動学)で説明できるか、接触の幾何(接触点のずれ・足面の向き)が必要か。

1. 前向きモデル(剛体の衝突 + 重力・空気抵抗・マグヌス力の飛翔):
   足(有効質量 M, 速度 V)が静止したボール(m=0.43kg, R=0.11m)に当たる。足の接触面は曲率半径 RHO_FOOT の曲面とみなし、ボール中心が足の軌跡から
   h だけずれると、接触法線は V の向きから asin(h/(R+RHO_FOOT)) だけ傾く。インパルスは法線方向(反発係数 e)+接触面の滑り方向(クーロン摩擦 μ、
   固着なら上限まで)で、ボールの出射 = J/m、スピン = (r×J)/I。入力を1つずつ動かしたときの着弾の変化(感度)を出す。
2. 被験者ごとの分解: 実測の着弾の共分散 Σ_obs を、足先速度の向きのばらつき(2Dの toe_vel_elev の SD)とスイング面モデル(Xsens の傾き τ:
   高く動くほど蹴り足の反対側へ y = -tanτ·z)で説明できる分 Σ_vel と、残り Σ_res に分け、残りの主軸の向き・大きさ(= 必要な接触点のずれ)を求める。
座標: 右手系(x 前, y 左, z 上)の右足キッカーで計算し、着弾は蹴り足側が + の y_k(= -y)で報告する(左足は鏡像で同じ意味)。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT
from .figio import save_figure

D = 11.0  # ゴールまでの距離 [m]
OUTLIER_ELEV_DEG = 15.0  # 足先速度の仰角の外れ値の閾値(中央値からの差)
G = 9.81
M_BALL, R_BALL = 0.43, 0.11
I_BALL = 2.0 / 3.0 * M_BALL * R_BALL**2
RHO_FOOT = 0.05  # 足の接触面の曲率半径 [m](仮定)
K_DRAG = 0.5 * 1.2 * 0.25 * np.pi * R_BALL**2 / M_BALL  # 空気抵抗 a = -K|v|v [1/m](Cd=0.25)
RHO_AIR_A_OVER_M = 0.5 * 1.2 * np.pi * R_BALL**2 / M_BALL
CL_MAX = 0.35


@dataclass(frozen=True)
class Params:
    speed: float = 20.0  # 足先速度 [m/s]
    elev: float = 3.0  # 足先速度の仰角 [deg]
    az: float = 0.0  # 足先速度の方位(+=蹴り足側) [deg]
    M: float = 1.4  # 有効質量 [kg]
    e: float = 0.6  # 反発係数
    mu: float = 0.4  # 摩擦係数
    h_up: float = 0.0  # ボール中心が足の軌跡より上 [m]
    h_lat: float = 0.0  # ボール中心が足の軌跡より蹴り足側 [m]


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def collide(p: Params) -> tuple[np.ndarray, np.ndarray]:
    """(ボールの初速 [m/s], 角速度 [rad/s])。右手系 x 前・y 左・z 上、蹴り足(右)側 = -y。"""
    e_, a_ = np.radians(p.elev), np.radians(p.az)
    vh = np.array([np.cos(e_) * np.cos(a_), -np.cos(e_) * np.sin(a_), np.sin(e_)])
    V = p.speed * vh
    up = _unit(np.array([0, 0, 1.0]) - vh[2] * vh)  # V に垂直な上
    kick = np.cross(vh, up)  # V に垂直な蹴り足側(右足なら -y 側)
    off = (p.h_up * up + p.h_lat * kick) / (R_BALL + RHO_FOOT)
    off = off * min(1.0, 0.95 / max(np.linalg.norm(off), 1e-9))
    n = _unit(vh * np.sqrt(1 - off @ off) + off)  # 足からボールへ向かう接触法線
    un = V @ n
    ut = V - un * n
    inv_m = 1 / p.M + 1 / M_BALL
    jn = (1 + p.e) * un / inv_m
    jt_stick = np.linalg.norm(ut) / (inv_m + R_BALL**2 / I_BALL)
    jt = min(p.mu * jn, jt_stick)
    J = jn * n + (jt * ut / np.linalg.norm(ut) if np.linalg.norm(ut) > 1e-9 else 0)
    return J / M_BALL, np.cross(-R_BALL * n, J) / I_BALL


def landing(v0: np.ndarray, w: np.ndarray, dt: float = 1e-3) -> tuple[float, float]:
    """ボール中心の原点から x=D に達する位置 (y_k, z) [m]。蹴り足側が +y_k。空気抵抗とマグヌス力あり。"""
    x, v = np.zeros(3), v0.copy()
    wn = np.linalg.norm(w)
    for _ in range(int(3 / dt)):
        sp = np.linalg.norm(v)
        a = -K_DRAG * sp * v + np.array([0, 0, -G])
        if wn > 1e-6:
            cl = min(CL_MAX, R_BALL * wn / sp)
            a = a + RHO_AIR_A_OVER_M * cl * sp**2 * _unit(np.cross(w, v))
        x_new, v = x + v * dt, v + a * dt
        if x_new[0] >= D:
            f = (D - x[0]) / (x_new[0] - x[0])
            xe = x + f * (x_new - x)
            return float(-xe[1]), float(xe[2])
        x = x_new
    return np.nan, np.nan


def land_of(p: Params) -> tuple[float, float]:
    return landing(*collide(p))


def aimed(base: Params, target_z: float = 1.0) -> Params:
    """目標(z=target_z)に着弾するよう、接触点の上下のずれ h_up を二分法で決める(足先速度の仰角は実測並みの値に固定)。"""
    lo, hi = -0.06, 0.06
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        z = land_of(replace(base, h_up=mid))[1]
        lo, hi = (mid, hi) if z < target_z else (lo, mid)
    return replace(base, h_up=0.5 * (lo + hi))


# (入力名, 刻み, 単位, 典型的なSD(感度を実際のばらつきに換算する仮定), 日本語名, 図の英語名)
INPUTS = [
    ("elev", 1.0, "deg", 2.5, "足先速度の仰角", "foot velocity elevation"),
    ("az", 1.0, "deg", 2.0, "足先速度の方位(蹴り足側)", "foot velocity azimuth (kick side)"),
    ("h_up", 0.01, "m", 0.015, "接触点のずれ: 上下", "contact offset, vertical"),
    ("h_lat", 0.01, "m", 0.015, "接触点のずれ: 左右(蹴り足側)", "contact offset, lateral (kick side)"),
    ("speed", 1.0, "m/s", 1.0, "足先速度の大きさ", "foot speed"),
    ("e", 0.05, "-", 0.05, "反発係数", "restitution e"),
    ("mu", 0.1, "-", 0.1, "摩擦係数", "friction mu"),
    ("M", 0.2, "kg", 0.2, "有効質量", "effective mass"),
]


def sensitivities(base: Params | None = None) -> pd.DataFrame:
    base = aimed(base or Params())
    y0, z0 = land_of(base)
    v0, w0 = collide(base)
    rows = []
    for name, step, unit, sd, label, label_en in INPUTS:
        y1, z1 = land_of(replace(base, **{name: getattr(base, name) + step}))
        v1, w1 = collide(replace(base, **{name: getattr(base, name) + step}))
        rows.append(dict(input=name, label=label, label_en=label_en, step=step, unit=unit, typical_sd=sd,
                         dy_k_m_per_step=y1 - y0, dz_m_per_step=z1 - z0,
                         dy_k_m_per_sd=(y1 - y0) / step * sd, dz_m_per_sd=(z1 - z0) / step * sd,
                         dspin_rpm_per_step=(np.linalg.norm(w1) - np.linalg.norm(w0)) * 60 / (2 * np.pi)))
    out = pd.DataFrame(rows)
    out.attrs.update(base=base, y0=y0, z0=z0, ball_speed=float(np.linalg.norm(v0)), spin_rpm=float(np.linalg.norm(w0) * 60 / (2 * np.pi)))
    return out


# ---------------- 被験者ごとの分解 ----------------
def _tilt(S: np.ndarray) -> float:
    w, V = np.linalg.eigh(S)
    v = V[:, np.argmax(w)]
    a = np.degrees(np.arctan2(v[1], v[0]))
    return float((a + 90) % 180 - 90)


def _robust_sd(x: pd.Series) -> float:
    x = x.dropna()
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else float("nan")


def decompose(df: pd.DataFrame, s_z_per_deg: float, tau_default: float, log=print) -> pd.DataFrame:
    rows = []
    for sess, g in df.dropna(subset=["land_z_m", "land_y_kickside_m", "toe_vel_elev_deg"]).groupby("session"):
        if len(g) < 5:
            continue
        S_obs = np.cov(np.vstack([g["land_y_kickside_m"], g["land_z_m"]]))
        names = g["trial"] if "trial" in g else pd.Series(g.index.astype(str), index=g.index)
        al = g["toe_vel_elev_deg"]
        bad = (al - al.median()).abs() > OUTLIER_ELEV_DEG  # インパクト検出のずれ・追跡失敗とみられる外れ値
        if bad.any():
            log(f"[{sess}] 足先速度の仰角が中央値±{OUTLIER_ELEV_DEG:g}°を超える試行を除いて計算(インパクト・追跡の確認が必要): "
                + ", ".join(f"{n}({v:+.0f}°)" for n, v in zip(names[bad], al[bad])))
        g = g[~bad]
        if len(g) < 5:
            continue
        S_obs = np.cov(np.vstack([g["land_y_kickside_m"], g["land_z_m"]]))
        has_tau = "xs_swing_axis_tilt_deg" in g and g["xs_swing_axis_tilt_deg"].notna().sum() >= 3
        tau = np.radians(g["xs_swing_axis_tilt_deg"].mean()) if has_tau else np.radians(tau_default)
        sd_alpha = _robust_sd(g["toe_vel_elev_deg"])  # 少数の追跡失敗に引きずられないよう MAD から求める
        sz = s_z_per_deg * sd_alpha  # 足先速度の仰角のばらつきが生む着弾 z の SD [m]
        t = -np.tan(tau)
        S_vel = sz**2 * np.array([[t * t, t], [t, 1.0]])  # (y_k, z): y = -tanτ·z
        S_res_raw = S_obs - S_vel
        w, V = np.linalg.eigh(S_res_raw)
        w_c = np.maximum(w, 0)
        S_res = V @ np.diag(w_c) @ V.T
        sd_major, sd_minor = np.sqrt(w_c[::-1])
        ang = np.degrees(np.arctan(sd_major / D))  # 残りの主軸方向のばらつきを角度に直す
        h_cm = (R_BALL + RHO_FOOT) * np.sin(np.radians(ang)) * 100  # それを作る接触点のずれ(曲率半径5cmの足)
        cd = g["contact_dy_m"] if "contact_dy_m" in g else pd.Series(dtype=float)
        h_meas = cd[(cd - cd.median()).abs() < 0.10].std() * 100 if len(cd) else np.nan  # 中央値から10cm以上のずれは追跡失敗として除く
        rows.append(dict(
            session=sess, n=len(g), tilt_obs_deg=_tilt(S_obs), sd_y_obs=np.sqrt(S_obs[0, 0]), sd_z_obs=np.sqrt(S_obs[1, 1]),
            toe_vel_elev_sd_deg=sd_alpha, tau_deg=np.degrees(tau), tau_from_xsens=int(has_tau),
            sd_z_from_velocity=sz, var_explained_by_velocity=float(min(1.0, np.trace(S_vel) / np.trace(S_obs))),
            tilt_velocity_model_deg=_tilt(S_vel), tilt_residual_deg=_tilt(S_res), resid_sd_major_m=sd_major, resid_sd_minor_m=sd_minor,
            resid_angle_deg=ang, resid_equiv_contact_offset_cm=h_cm, measured_contact_dy_sd_cm=h_meas,
            resid_neg_eig_clipped=int(w.min() < 0)))
        r = rows[-1]
        log(f"[{sess}] 実測の傾き {r['tilt_obs_deg']:+.0f}° (SD y {r['sd_y_obs']:.2f} / z {r['sd_z_obs']:.2f} m) | 足先速度仰角 SD {sd_alpha:.1f}° → z の SD {sz:.2f} m "
            f"(分散の{r['var_explained_by_velocity'] * 100:.0f}%、傾き{r['tilt_velocity_model_deg']:+.0f}°, τ={np.degrees(tau):+.0f}°{'' if has_tau else '(仮定)'}) | "
            f"残り: 主軸の傾き {r['tilt_residual_deg']:+.0f}°, SD {sd_major:.2f}/{sd_minor:.2f} m = {ang:.1f}° = 接触点のずれ約{h_cm:.1f} cm"
            f"(2Dで測った上下のずれの SD {h_meas:.1f} cm)")
    return pd.DataFrame(rows)


# ---------------- 図 ----------------
def plot_tornado(sens: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    lab = [f"{r.label_en} (+/-{r.typical_sd:g} {r.unit})" for r in sens.itertuples()]
    y = np.arange(len(sens))[::-1]
    for ax, col, ttl in ((axes[0], "dz_m_per_sd", "vertical z at the goal (m)"), (axes[1], "dy_k_m_per_sd", "lateral y, kick side + (m)")):
        ax.barh(y, sens[col], color="#2a78d6")
        ax.axvline(0, color="#1a1a19", lw=0.8)
        ax.set_title(f"effect of +1 typical SD on {ttl}", fontsize=8)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(lab, fontsize=7)
    b = sens.attrs["base"]
    fig.suptitle(f"Contact-model sensitivity (baseline aimed at the target by contact offset {b.h_up * 100:.1f} cm: foot {b.speed:g} m/s, elev {b.elev:.1f} deg, M={b.M:g} kg, e={b.e:g}, mu={b.mu:g}; ball {sens.attrs['ball_speed']:.1f} m/s, spin {sens.attrs['spin_rpm']:.0f} rpm)", fontsize=7)
    fig.tight_layout()
    save_figure(fig, path, dpi=140)
    plt.close(fig)


def plot_tilt_map(dec: pd.DataFrame, s_z_per_deg: float, tau_deg: float, path: Path) -> None:
    """速度方向のばらつき(傾き -70°付近の細長い楕円)と接触点のずれ(主軸方向 φ)の分散の割合で、着弾の楕円の傾きがどう変わるか。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = -np.tan(np.radians(tau_deg))
    u = np.array([t, 1.0]) / np.hypot(t, 1.0)
    fig, ax = plt.subplots(figsize=(6, 4))
    share = np.linspace(0, 1, 101)
    for phi, col in ((0, "#6b6b66"), (20, "#1baf7a"), (45, "#eb6834"), (70, "#e87ba4")):
        c = np.array([np.cos(np.radians(phi)), np.sin(np.radians(phi))])
        tilts = [_tilt(s * np.outer(u, u) + (1 - s) * (np.outer(c, c) * 0.85 + 0.15 * np.eye(2) / 1.0)) for s in share]
        ax.plot(share * 100, tilts, color=col, label=f"contact-offset axis {phi} deg")
    for r in dec.itertuples():
        ax.axhline(r.tilt_obs_deg, color="#2a78d6", lw=0.8, ls="--")
        ax.text(101, r.tilt_obs_deg, r.session, fontsize=7, va="center", color="#2a78d6")
    ax.set_xlabel("share of landing variance from foot-velocity direction (%)", fontsize=8)
    ax.set_ylabel("tilt of landing ellipse (deg)", fontsize=8)
    ax.set_title(f"Ellipse tilt from two sources (swing-plane tilt {tau_deg:.0f} deg) vs observed (dashed)", fontsize=8)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, path, dpi=140)
    plt.close(fig)


def run_contact(root: Path | None = None, log=print) -> pd.DataFrame:
    root = root or ROOT / "output"
    tdir, fdir = root / "analysis" / "tables", root / "analysis" / "landing"
    tdir.mkdir(parents=True, exist_ok=True)
    sens = sensitivities()
    b = sens.attrs["base"]
    log(f"基準(足先速度の仰角{b.elev:.0f}°のまま、目標に着弾するよう接触点の上下のずれを {b.h_up * 100:.1f} cm に調整): 足先 {b.speed:g} m/s, M={b.M:g} kg, e={b.e:g}, μ={b.mu:g} → ボール {sens.attrs['ball_speed']:.1f} m/s, スピン {sens.attrs['spin_rpm']:.0f} rpm")
    for r in sens.itertuples():
        log(f"  {r.label:<18} +{r.step:g}{r.unit}: 着弾 z {r.dz_m_per_step:+.3f} m, y(蹴り足側) {r.dy_k_m_per_step:+.3f} m"
            f"   | 典型SD ±{r.typical_sd:g}{r.unit} なら z {r.dz_m_per_sd:+.2f} m, y {r.dy_k_m_per_sd:+.2f} m")
    sens.to_csv(tdir / "contact_sensitivity.csv", index=False, float_format="%.4f")
    try:
        plot_tornado(sens, fdir / "contact_sensitivity.png")
    except OSError as e:
        log(f"[警告] 図を保存できません: {e}")
    s_z_per_deg = float(sens.loc[sens["input"] == "elev", "dz_m_per_step"].iloc[0])  # 足先速度の仰角1°あたりの着弾 z の変化 [m]
    p = tdir / "kick_factors.csv"
    if not p.exists():
        log("kick_factors.csv がないため被験者ごとの分解は省略(先に python run.py factors)")
        return sens
    df = pd.read_csv(p)
    if "land_z_m" not in df:
        log("着弾(config/landing.csv)が結合されていないため被験者ごとの分解は省略")
        return sens
    tau_all = df["xs_swing_axis_tilt_deg"].mean() if "xs_swing_axis_tilt_deg" in df and df["xs_swing_axis_tilt_deg"].notna().any() else 20.0
    dec = decompose(df, s_z_per_deg, tau_all, log)
    if not dec.empty:
        dec.to_csv(tdir / "contact_decomposition.csv", index=False, float_format="%.3f")
        log(f"保存: {tdir / 'contact_decomposition.csv'}")
        try:
            plot_tilt_map(dec, s_z_per_deg, tau_all, fdir / "contact_tilt_map.png")
        except OSError as e:
            log(f"[警告] 図を保存できません: {e}")
    return sens
