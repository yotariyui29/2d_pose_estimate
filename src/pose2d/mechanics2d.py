"""2次元運動学(kinematics.csv, m単位)から力学量を算出する。

定義は 3d_pose_estimate/src/analysis/energy_flow.py(SI2025準拠)に揃えている:
  - セグメント: 阿江ら(1992)の身体部分慣性係数。体幹は左右半分ずつ、頭は頭頂-首の45%点
  - E_seg = 並進KE + 回転KE + PE(重心高=キャリブ原点からの高さ y)
  - W_p = インパクト前0.1sにおける蹴り脚(大腿+下腿+足)の Σ|dE_seg|   (全身版は W_p_body)
  - E_b = 1/2 m_ball v_ball^2, v_ball = インパクト後 0.01〜0.02 s のボール位置の直線近似の速さ
  - EI = E_b / W_p
座標は蹴り方向が +x になるよう必要なら x を反転する(被験者・蹴り足によらず前方が+x)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

G = 9.80665
BALL_MASS_KG = 0.43
WORK_WINDOW_S = 0.1
BALL_WINDOW_S = (0.01, 0.02)
M_EFF_CAP_KG = 30.0
HAND_TO_FOREARM = 0.36  # 手/前腕の長さ比(手のマーカー無し)

# 名前: (近位, 遠位, 質量比, 近位からの重心比, 重心まわりの回転半径比)
LIMBS = {
    "upperarm": ("shoulder", "elbow", 0.027, 0.529, 0.262),
    "forearm": ("elbow", "wrist", 0.022, 0.415, 0.279),
    "thigh": ("hip", "knee", 0.110, 0.475, 0.278),
    "shank": ("knee", "ankle", 0.051, 0.406, 0.274),
}
SIDES = {"L": "left", "R": "right"}


def _p(k: pd.DataFrame, name: str, sgn: float) -> np.ndarray | None:
    if f"{name}_x" not in k:
        return None
    return np.column_stack([sgn * k[f"{name}_x"].to_numpy(), k[f"{name}_y"].to_numpy()])


def kick_sign(k: pd.DataFrame, kick_side: str) -> float:
    """インパクト直前0.05sの足先(母趾)速度のx成分の符号 → 蹴り方向を+xにする係数。"""
    col = f"{kick_side}_big_toe_vx"
    if col not in k or "t_rel_impact_s" not in k:
        return 1.0
    sel = k["t_rel_impact_s"].between(-0.05, 0.0)
    v = k.loc[sel, col].mean()
    return -1.0 if np.isfinite(v) and v < 0 else 1.0


def segments(k: pd.DataFrame, sgn: float) -> tuple[dict[str, dict], list[str]]:
    """{name: {m, com, prox, dist, rho}} (m: 質量比)。使えなかった部位/推定した部位のメモも返す。"""
    seg: dict[str, dict] = {}
    notes: list[str] = []
    neck, pelvis = _p(k, "neck", sgn), _p(k, "pelvis", sgn)
    for S, side in SIDES.items():
        for name, (a, b, m, r, rho) in LIMBS.items():
            pa, pb = _p(k, f"{side}_{a}", sgn), _p(k, f"{side}_{b}", sgn)
            if pa is None or pb is None:
                continue
            seg[f"{S}{name}"] = dict(m=m, com=pa + r * (pb - pa), prox=pa, dist=pb, rho=rho)
        if f"{S}forearm" in seg:  # 前腕+手を1セグメントとして重心を手側へ寄せる
            el, wr = _p(k, f"{side}_elbow", sgn), _p(k, f"{side}_wrist", sgn)
            hand = wr + 0.891 * HAND_TO_FOREARM * (wr - el)
            seg[f"{S}forearm"]["com"] = (0.016 * (el + 0.415 * (wr - el)) + 0.006 * hand) / 0.022
        heel, big, small, ank = (_p(k, f"{side}_{n}", sgn) for n in ("heel", "big_toe", "small_toe", "ankle"))
        if heel is not None and big is not None and ank is not None:
            toe = (big + small) / 2 if small is not None else big
            seg[f"{S}foot"] = dict(m=0.011, com=toe + 0.595 * (heel - toe), prox=ank, dist=toe, rho=0.177, heel=heel)
        sh = _p(k, f"{side}_shoulder", sgn)
        if sh is not None and pelvis is not None:
            seg[f"{S}torso"] = dict(m=0.2445, com=sh + 0.493 * (pelvis - sh), prox=sh, dist=pelvis, rho=0.346)
    head = _p(k, "head", sgn)
    if neck is not None and pelvis is not None:
        if head is None or not np.isfinite(head).any():
            head = neck + 0.45 * (neck - pelvis)  # 頭頂が未検出(古いraw_px)の場合は体幹方向から推定
            notes.append("head推定")
        seg["head"] = dict(m=0.069, com=head + 0.45 * (neck - head), prox=head, dist=neck, rho=0.3)
    return seg, notes


def _unwrap(a: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan)
    ok = np.isfinite(a)
    out[ok] = np.unwrap(a[ok])
    return out


def energies(seg: dict[str, dict], mass_kg: float, dt: float) -> dict[str, dict]:
    res = {}
    for name, s in seg.items():
        v = np.gradient(s["com"], dt, axis=0)
        axis = s["dist"] - s["prox"]
        omega = np.gradient(_unwrap(np.arctan2(axis[:, 1], axis[:, 0])), dt)
        m = s["m"] * mass_kg
        inertia = m * (axis**2).sum(axis=1) * s["rho"] ** 2
        ke = 0.5 * m * (v**2).sum(axis=1) + 0.5 * inertia * omega**2
        res[name] = dict(m=m, v=v, omega=omega, ke=ke, total=ke + m * G * s["com"][:, 1], inertia=inertia)
    return res


def effective_mass(seg: dict[str, dict], en: dict[str, dict], kick: str, dt: float) -> np.ndarray:
    """蹴り脚(股関節固定の平面3リンク: 大腿-下腿-足)のKhatib有効質量 [kg]。
    作用点=足の踵-母趾の中点、方向=作用点の速度方向。1/(u^T J M^-1 J^T u)。"""
    th, sh, ft = seg[f"{kick}thigh"], seg[f"{kick}shank"], seg[f"{kick}foot"]
    hip, knee, ank = th["prox"], th["dist"], sh["dist"]
    contact = 0.5 * (ft["heel"] + ft["dist"])
    u = np.gradient(contact, dt, axis=0)
    spd = np.hypot(u[:, 0], u[:, 1])
    ms = [en[f"{kick}{n}"]["m"] for n in ("thigh", "shank", "foot")]
    inertia = [en[f"{kick}{n}"]["inertia"] for n in ("thigh", "shank", "foot")]

    def perp(v):
        return np.stack([-v[..., 1], v[..., 0]], axis=-1)

    j1k, j2a = perp(knee - hip), perp(ank - knee)
    zero = np.zeros_like(j1k)
    jc = [np.stack([perp(th["com"] - hip), zero, zero], -1),
          np.stack([j1k, perp(sh["com"] - knee), zero], -1),
          np.stack([j1k, j2a, perp(ft["com"] - ank)], -1)]  # (N, 2, 3)
    jt = np.stack([j1k, j2a, perp(contact - ank)], -1)
    out = np.full(len(hip), np.nan)
    for i in range(len(hip)):
        if spd[i] < 0.5 or not np.isfinite(jt[i]).all() or not all(np.isfinite(j[i]).all() for j in jc):
            continue
        M = sum(m * j[i].T @ j[i] for m, j in zip(ms, jc)) + np.diag([q[i] for q in inertia])
        try:
            lam_inv = jt[i] @ np.linalg.inv(M) @ jt[i].T
        except np.linalg.LinAlgError:
            continue
        d = u[i] / spd[i]
        den = d @ lam_inv @ d
        out[i] = min(M_EFF_CAP_KG, 1.0 / den) if den > 1e-9 else M_EFF_CAP_KG
    return out


def ball_speed(k: pd.DataFrame) -> float:
    """インパクト後 0.01〜0.02 s のボール位置を直線近似した速さ [m/s]。"""
    if "ball_x" not in k or "t_rel_impact_s" not in k:
        return np.nan
    t = k["t_rel_impact_s"].to_numpy()
    sel = (t >= BALL_WINDOW_S[0] - 1e-9) & (t <= BALL_WINDOW_S[1] + 1e-9)
    p = k.loc[sel, ["ball_x", "ball_y"]].to_numpy()
    ok = np.isfinite(p).all(1)
    if ok.sum() < 3:
        return np.nan
    return float(np.hypot(*[np.polyfit(t[sel][ok], p[ok, i], 1)[0] for i in range(2)]))


def analyze_trial(k: pd.DataFrame, info: dict) -> tuple[dict, pd.DataFrame]:
    """1試行の指標(dict)と、インパクト基準の時系列(DataFrame)を返す。"""
    side = info["kick_side"]
    S = side[0].upper()
    sup = "R" if S == "L" else "L"
    fps = float(info["fps"])
    dt = 1.0 / fps
    mass = float(info.get("mass_kg") or 60.0)
    sgn = kick_sign(k, side)
    seg, notes = segments(k, sgn)
    en = energies(seg, mass, dt)

    t = k["t_rel_impact_s"].to_numpy()
    i1 = int(np.nanargmin(np.abs(t)))
    i0 = i1 - int(round(WORK_WINDOW_S * fps))
    ts = pd.DataFrame({"t": t})

    mtot = sum(e["m"] for e in en.values())
    vcom = sum(e["m"] * e["v"] for e in en.values()) / mtot
    ts["momentum_x"], ts["momentum_y"] = mtot * vcom[:, 0], mtot * vcom[:, 1]
    ts["body_com_x"], ts["body_com_y"] = (sum(e["m"] * s["com"] for e, s in zip(en.values(), seg.values())) / mtot).T
    for key, name in (("kick_torso", f"{S}torso"), ("kick_thigh", f"{S}thigh"), ("kick_shank", f"{S}shank"),
                      ("support_torso", f"{sup}torso"), ("support_thigh", f"{sup}thigh"), ("support_shank", f"{sup}shank")):
        ts[f"E_{key}"] = en[name]["ke"] if name in en else np.nan
    for n in ("thigh", "shank", "foot"):
        ts[f"omega_{n}"] = en[f"{S}{n}"]["omega"] if f"{S}{n}" in en else np.nan  # rad/s、前方スイングが正
    ts["toe_speed"] = k.get(f"{side}_big_toe_speed", np.nan)
    ts["ankle_speed"] = k.get(f"{side}_ankle_speed", np.nan)
    ts["knee_flexion"] = k.get(f"knee_flexion_{S}_deg", np.nan)
    ts["m_eff"] = effective_mass(seg, en, S, dt) if all(f"{S}{n}" in seg for n in ("thigh", "shank", "foot")) else np.nan

    leg = [f"{S}thigh", f"{S}shank", f"{S}foot"]
    work = float(sum(np.abs(np.diff(en[n]["total"][i0:i1 + 1])).sum() for n in leg if n in en))
    work_body = float(sum(np.abs(np.diff(e["total"][i0:i1 + 1])).sum() for e in en.values()))
    net = float(sum(en[n]["total"][i1] - en[n]["total"][i0] for n in leg if n in en))
    v_ball = ball_speed(k)
    e_b = 0.5 * BALL_MASS_KG * v_ball**2 if np.isfinite(v_ball) else np.nan

    swing = (t >= -0.25) & (t <= 0.02)
    pk = {n: (t[swing][np.nanargmax(ts[f"omega_{n}"].to_numpy()[swing])] if np.isfinite(ts[f"omega_{n}"][swing]).any() else np.nan)
          for n in ("thigh", "shank", "foot")}
    toe = ts["toe_speed"].to_numpy()
    m_eff_pre = ts.loc[(t >= -0.01) & (t <= 0.0), "m_eff"]
    metrics = dict(
        impact_frame=info.get("impact_frame"), fps=fps, mass_kg=mass,
        ball_speed_mps=v_ball, ball_speed_kmh=v_ball * 3.6, E_b_J=e_b,
        W_p_J=work, W_p_body_J=work_body, W_p_net_J=net, EI=e_b / work if work > 0 else np.nan,
        toe_speed_impact_mps=toe[i1], toe_speed_peak_mps=np.nanmax(toe[swing]),
        toe_speed_peak_t_s=t[swing][np.nanargmax(toe[swing])],
        shank_omega_peak_rads=np.nanmax(ts["omega_shank"].to_numpy()[swing]),
        ball_foot_ratio=v_ball / toe[i1] if toe[i1] > 0 else np.nan,
        m_eff_impact_kg=float(ts["m_eff"].iloc[i1]), m_eff_pre10ms_kg=float(m_eff_pre.mean()) if m_eff_pre.notna().any() else np.nan,
        knee_flexion_impact_deg=float(ts["knee_flexion"].iloc[i1]),
        peak_t_thigh_s=pk["thigh"], peak_t_shank_s=pk["shank"], peak_t_foot_s=pk["foot"],
        momentum_x_impact=float(ts["momentum_x"].iloc[i1]), x_flipped=int(sgn < 0), notes=";".join(notes),
    )
    return metrics, ts
