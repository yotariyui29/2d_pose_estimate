"""高速度カメラ2D + Xsens姿勢(stop基準) による胴体以下の3D運動学。

方法:
  Rel_i(t) = R_i(t) R_i,stop^-1  … 直立(stop)姿勢からのセグメント回転（世界系G）
  セグメント方向 = Rel_i(t)·v0（v0: 直立時のセグメントベクトル。大腿・下腿は鉛直下向き、足部は前方）
  補正(refine_heading): センサーごとに Q = Rz(-a)·Rel·P（a=試行中のヘディングずれ, P=stop姿勢の中立からのずれ）
  未知量: カメラ水平方位 phi（Gでの画像x軸）、直立時の正面方位 psi、時間ずれ lag、a・P
        → キック脚の大腿・下腿・足部の2D画像内角度との一致、膝・股・足首の外転・回旋の小ささで推定
  3D位置: キック脚の股関節=2D位置(奥行き=0)、反対側の股関節=骨盤姿勢×股幅、以降 順運動学。セグメント長は2Dの95%タイル。
座標系W: x=画像右, y=奥行き(z×x), z=鉛直上。単位m。
Usage: python src/imu_video_3d.py --trial kurogi_t1 [--source auto|pose2d|kinovea] [--no-align]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R, Slerp

from pose2d import xsensfactors as xsf
from pose2d.figio import save_figure
from .trials import ROOT, WINDOW, Trial, Trial2D, load_trials

OUT = ROOT / "output" / "imu3d"
FS = 100.0  # Xsensのサンプリング周波数[Hz]
HIP_WIDTH = 0.18  # m（仮定。2Dからは求まらない）
Z = np.array([0.0, 0.0, 1.0])
DOWN = -Z
ALIGN_KEYS = ["up_L", "leg_L", "foot_L", "up_R", "leg_R", "foot_R"]  # 腰は基準(a=0)
MIN_WINDOW_S = 0.3


def analysis_window(t2: Trial2D) -> np.ndarray:
    lo, hi = max(WINDOW[0], t2.t.min()), min(WINDOW[1] - 1 / FS, t2.t.max())
    return np.arange(np.ceil(lo * FS), np.floor(hi * FS) + 1) / FS


def _phase_dirs(tr: Trial) -> tuple[Path, Path]:
    """(試行の記録フォルダ, 直立(stop)フォルダ)。task/ が無い計測は、フォルダ直下が試行の記録。"""
    task = tr.xsens_dir / "task"
    return (task if task.is_dir() else tr.xsens_dir), tr.xsens_dir / "stop"


def rel_rotations(tr: Trial) -> dict[str, R]:
    """直立姿勢からの回転（世界系）= R(t)·R_stop^-1。
    姿勢は Quat があればそれを、無ければ Roll/Pitch/Yaw を使い、フレーム欠落は補間して100Hzに揃える(pose2d.xsensfactors)。
    stop が無い、または stop に姿勢が出力されていない計測は、試行中で最も静止した0.5秒を直立とみなす。"""
    task_dir, stop_dir = _phase_dirs(tr)
    rot: dict[str, R] = {}
    for k, sid in tr.sensors.items():
        got = xsf._load(task_dir, sid)
        if got is None:
            raise FileNotFoundError(f"センサー {k}({sid}) の記録がありません: {task_dir}")
        rot[k] = got[0]
    n = min(len(v) for v in rot.values())
    rot = {k: v[:n] for k, v in rot.items()}
    ref: dict[str, R] | None = None
    if stop_dir.is_dir():
        try:
            ref = {k: xsf._load(stop_dir, sid)[0].mean() for k, sid in tr.sensors.items()}
        except (KeyError, ValueError, TypeError):
            ref = None
    if ref is None:
        w = int(0.5 * FS)
        tot = sum(np.linalg.norm(xsf._omega_world(v), axis=1) for v in rot.values())
        s0 = int(np.argmin(np.convolve(tot, np.ones(w) / w, mode="valid")))
        ref = {k: v[s0:s0 + w].mean() for k, v in rot.items()}
    return {k: rot[k] * ref[k].inv() for k in rot}


def sample(rot: R, t: np.ndarray) -> R:
    tt = np.arange(len(rot)) / FS
    return Slerp(tt, rot)(np.clip(t, 0, tt[-1]))


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def ang_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2Dベクトル列a,bのなす角（rad, 符号付き）。"""
    return np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(1))


def img_vec(d: np.ndarray, phi: float) -> np.ndarray:
    ex = np.array([np.cos(phi), np.sin(phi), 0.0])
    return np.stack([d @ ex, d @ Z], 1)


def dirs_2d(p: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {"thigh": unit(p["KNEE"] - p["HIP"]), "shank": unit(p["ANKLE"] - p["KNEE"]), "foot": unit(p["MP"] - p["ANKLE"])}


def foot0(psi: float, delta: float) -> np.ndarray:
    """直立時の足部方向（正面方位psi、水平から下向きδ）。"""
    return np.array([np.cos(psi) * np.cos(delta), np.sin(psi) * np.cos(delta), -np.sin(delta)])


class Fit:
    """1試行の当てはめコンテキスト。"""

    def __init__(self, tr: Trial, t2: Trial2D, rel: dict[str, R]):
        self.tr, self.t2, self.rel, self.K = tr, t2, rel, tr.kick
        self.tau = analysis_window(t2)
        self.t_imp = tr.impact_s
        self.d2 = dirs_2d(t2.at(self.tau))
        self.delta = t2.foot_pitch()
        self.L = t2.lengths()

    def key(self, seg: str) -> str:
        return f"{seg}_{self.K}"

    def fit_phi_lag(self):
        """大腿・下腿の画像内角度で phi, lag を粗探索→精密化。返値 (rms rad, phi, lag)。"""
        cache: dict[float, list[np.ndarray]] = {}

        def dirs(lag: float) -> list[np.ndarray]:
            k = round(lag, 4)
            if k not in cache:
                cache[k] = [sample(self.rel[self.key(s)], self.t_imp + lag + self.tau).apply(DOWN) for s in ("up", "leg")]
            return cache[k]

        def cost(phi: float, lag: float) -> float:
            e = [ang_diff(img_vec(d, phi), self.d2[s]) for d, s in zip(dirs(lag), ("thigh", "shank"))]
            return float(np.sqrt(np.mean(np.concatenate(e) ** 2)))

        best = (1e9, 0.0, 0.0)
        for lag in np.arange(-0.10, 0.101, 0.02):
            for phi in np.radians(np.arange(0, 360, 4)):
                c = cost(phi, lag)
                if c < best[0]:
                    best = (c, phi, lag)
        _, p0, l0 = best
        for lag in np.arange(l0 - 0.02, l0 + 0.0201, 0.005):
            for phi in p0 + np.radians(np.arange(-4, 4.01, 0.5)):
                c = cost(phi, lag)
                if c < best[0]:
                    best = (c, phi, lag)
        return best

    def fit_psi(self, phi: float, lag: float):
        rf = sample(self.rel[self.key("foot")], self.t_imp + lag + self.tau)
        best = (1e9, 0.0)
        for psi in np.radians(np.arange(0, 360, 1.0)):
            c = float(np.sqrt(np.mean(ang_diff(img_vec(rf.apply(foot0(psi, self.delta)), phi), self.d2["foot"]) ** 2)))
            if c < best[0]:
                best = (c, psi)
        return best

    def refine_heading(self, phi0: float, psi0: float, lag0: float, w: dict[str, float] | None = None, sel: np.ndarray | None = None):
        """センサーごとのヘディングずれ a(6) とP(6x3)、phi, psi, lag を同時推定。
        残差: 2D画像内の大腿・下腿・足部角度 ＋ 外転・回旋の小ささ（膝を強く、股・足首は弱く）＋ a,Pの正則化。
        sel: 当てはめに使う時刻のマスク（検証用）。返値 (cost, x=[phi, psi, a*6, P*18], lag)。"""
        w = {"knee": 1.0, "hip": 0.3, "ankle": 0.2, "ridge": 0.02, "pose": 0.5, **(w or {})}
        sel = np.ones(len(self.tau), bool) if sel is None else sel
        d2 = {k: v[sel] for k, v in self.d2.items()}
        n = int(sel.sum())
        samp: dict[float, dict[str, R]] = {}

        def S(lag: float) -> dict[str, R]:
            k = round(lag, 4)
            if k not in samp:
                samp[k] = {key: sample(v, self.t_imp + lag + self.tau[sel]) for key, v in self.rel.items()}
            return samp[k]

        def resid(x: np.ndarray, lag: float) -> np.ndarray:
            phi, psi, a, p = x[0], x[1], x[2:8], x[8:].reshape(6, 3)
            r = yaw_correct(S(lag), a, p)
            e = [ang_diff(img_vec(r[self.key("up")].apply(DOWN), phi), d2["thigh"]),
                 ang_diff(img_vec(r[self.key("leg")].apply(DOWN), phi), d2["shank"]),
                 ang_diff(img_vec(r[self.key("foot")].apply(foot0(psi, self.delta)), phi), d2["foot"])]
            for key, arr in joint_offplane(r, psi).items():
                e.append(w[key[:-1]] * np.radians(arr).ravel())
            e.append(w["ridge"] * np.sqrt(n) * a)
            e.append(w["pose"] * np.sqrt(n) * p.ravel())
            return np.concatenate(e) / np.sqrt(n)

        best = (1e9, None, None)
        for lag in lag0 + np.arange(-0.01, 0.0101, 0.005):
            for dphi in np.radians((-25, 0, 25)):
                x0 = np.concatenate([[phi0 + dphi, psi0], np.zeros(24)])
                sol = least_squares(resid, x0, args=(lag,), x_scale=0.2)
                if sol.cost < best[0]:
                    best = (sol.cost, sol.x, lag)
        return best

    def metrics(self, rel: dict[str, R], phi: float, psi: float, lag: float) -> dict[str, float]:
        rr = {k: sample(v, self.t_imp + lag + self.tau) for k, v in rel.items()}
        m = {f"offplane_{k}": float(np.sqrt(np.mean(v ** 2))) for k, v in joint_offplane(rr, psi).items()}
        for key, s in (("up", "thigh"), ("leg", "shank")):
            m[f"rms2d_{s}"] = float(np.degrees(np.sqrt(np.mean(ang_diff(img_vec(rr[self.key(key)].apply(DOWN), phi), self.d2[s]) ** 2))))
        return m


def yaw_correct(rot: dict[str, R], a: np.ndarray, p: np.ndarray | None = None) -> dict[str, R]:
    """センサー姿勢の補正: Q = Rz(-a)·Rel·P。a: 試行中のヘディングずれ[rad]（鉛直軸まわり）、
    P(rotvec, 6x3): stop直立姿勢が中立からずれている分（初期傾き・ねじれ）。"""
    out = dict(rot)
    for i, (k, ai) in enumerate(zip(ALIGN_KEYS, a)):
        out[k] = R.from_euler("z", -ai) * rot[k] * (R.from_rotvec(p[i]) if p is not None else R.identity())
    return out


def joint_angles(rp: R, rc: R, psi: float, side: str) -> np.ndarray:
    """親子の相対回転を正面方位基準で YXZ 分解 → [Y, 外転(+), 内旋(+)] deg。屈曲は呼び出し側で符号調整（股=-Y、膝/底屈=+Y）。"""
    H = R.from_euler("z", psi)
    y, x, z = np.degrees((H.inv() * rp.inv() * rc * H).as_euler("YXZ").T)
    s = 1.0 if side == "L" else -1.0
    return np.stack([y, s * x, s * z], 1)


def joint_offplane(r: dict[str, R], psi: float) -> dict[str, np.ndarray]:
    """各関節の外転・回旋成分 [n,2] (deg)。"""
    out = {}
    for s in "LR":
        for jn, (p, c) in {"hip": ("pelvis", f"up_{s}"), "knee": (f"up_{s}", f"leg_{s}"), "ankle": (f"leg_{s}", f"foot_{s}")}.items():
            out[f"{jn}{s}"] = joint_angles(r[p], r[c], psi, s)[:, 1:]
    return out


def to_world(v: np.ndarray, phi: float) -> np.ndarray:
    """G → W（x=画像右, y=奥行き, z=鉛直）。"""
    ex = np.array([np.cos(phi), np.sin(phi), 0.0])
    return np.stack([v @ ex, v @ np.cross(Z, ex), v @ Z], -1)


def run(tr: Trial, source: str = "auto", align: bool = True, out_dir: Path | None = None, verbose: bool = True):
    """1試行を処理。返値 (df, ctx, summary)。2Dが無い/窓が短い試行は None を返す。"""
    log = print if verbose else (lambda *a, **k: None)
    t2 = tr.load_2d(source)
    if t2 is None:
        log(f"[{tr.name}] 2Dデータなし（source={source}）→ スキップ")
        return None
    if any(np.isnan(v).any() for v in t2.pts.values()):
        log(f"[{tr.name}] 2Dに補間しきれない欠けがある → スキップ")
        return None
    c = Fit(tr, t2, rel_rotations(tr))
    if len(c.tau) / FS < MIN_WINDOW_S:
        log(f"[{tr.name}] 解析区間が短い({len(c.tau) / FS:.2f}s) → スキップ")
        return None
    log(f"[{tr.name}] 2D={t2.source} 区間 {c.tau[0]:+.2f}〜{c.tau[-1]:+.2f}s ({len(c.tau)}点) キック脚={c.K}")
    cost0, phi, lag = c.fit_phi_lag()
    cpsi, psi = c.fit_psi(phi, lag)
    log(f"  初期: phi={np.degrees(phi) % 360:.1f}° lag={lag * 1000:+.0f}ms rms(大腿,下腿)={np.degrees(cost0):.1f}° | psi={np.degrees(psi) % 360:.1f}° rms(足部)={np.degrees(cpsi):.1f}°"
        f" | L={ {k: round(v, 3) for k, v in c.L.items()} }")
    rel = c.rel
    before = c.metrics(rel, phi, psi, lag)
    a_deg, p_deg = np.zeros(6), np.zeros(6)
    if align:
        _, x, lag = c.refine_heading(phi, psi, lag)
        phi, psi, a, pp = x[0], x[1], x[2:8], x[8:].reshape(6, 3)
        rel = yaw_correct(rel, a, pp)
        a_deg, p_deg = np.degrees(a), np.degrees(np.linalg.norm(pp, axis=1))
        log("  ヘディング補正[deg]: " + " ".join(f"{k}={v:+.1f}" for k, v in zip(ALIGN_KEYS, a_deg)) +
            " | |P|: " + " ".join(f"{k}={v:.1f}" for k, v in zip(ALIGN_KEYS, p_deg)) +
            f" | phi={np.degrees(phi) % 360:.1f} psi={np.degrees(psi) % 360:.1f} lag={lag * 1000:+.0f}ms")
    after = c.metrics(rel, phi, psi, lag)
    log("  外転・回旋RMS(deg) 補正前→後: " + " ".join(f"{k[9:]}={before[k]:.1f}→{after[k]:.1f}" for k in before if k.startswith("offplane")))

    tau = c.tau
    t = c.t_imp + lag + tau
    r = {k: sample(v, t) for k, v in rel.items()}
    p2 = t2.at(tau)
    f0 = c.L["foot"] * foot0(psi, c.delta)
    left_axis = np.array([-np.sin(psi), np.cos(psi), 0.0])
    K, O = c.K, ("R" if c.K == "L" else "L")
    ex = np.array([np.cos(phi), np.sin(phi), 0.0])
    hipK = p2["HIP"][:, 0:1] * ex + p2["HIP"][:, 1:2] * Z  # G座標（奥行き=0）
    lat = HIP_WIDTH * r["pelvis"].apply(left_axis)  # 骨盤の左向き
    pos = {f"hip{K}": hipK, f"hip{O}": hipK - lat if K == "L" else hipK + lat}
    for s in ("L", "R"):
        pos[f"knee{s}"] = pos[f"hip{s}"] + c.L["thigh"] * r[f"up_{s}"].apply(DOWN)
        pos[f"ankle{s}"] = pos[f"knee{s}"] + c.L["shank"] * r[f"leg_{s}"].apply(DOWN)
        pos[f"toe{s}"] = pos[f"ankle{s}"] + r[f"foot_{s}"].apply(f0)
    df = pd.DataFrame({"tau": tau, "t_xsens": t})
    for k, v in pos.items():
        w = to_world(v, phi)
        for i, ax in enumerate("xyz"):
            df[f"{k}_{ax}"] = w[:, i]
    for s in ("L", "R"):
        for jn, (p, ch) in {"hip": (r["pelvis"], r[f"up_{s}"]), "knee": (r[f"up_{s}"], r[f"leg_{s}"]),
                            "ankle": (r[f"leg_{s}"], r[f"foot_{s}"])}.items():
            a_ = joint_angles(p, ch, psi, s)
            if jn == "hip":
                a_[:, 0] = -a_[:, 0]
            for i, nm in enumerate(("flex", "abd", "rot")):
                df[f"{jn}{s}_{nm}"] = a_[:, i]
    H = R.from_euler("z", psi)
    df["pelvis_yaw"], df["pelvis_pitch"], df["pelvis_roll"] = (H.inv() * r["pelvis"] * H).as_euler("ZYX", degrees=True).T

    # 検証: キック脚の2Dマーカーとの位置誤差（股を2Dに固定しているため膝・足首・つま先で評価）
    err = {n2: float(np.sqrt(np.mean(((np.stack([df[f"{n3}{K}_x"], df[f"{n3}{K}_z"]], 1) - p2[n2]) ** 2).sum(1))) * 1000)
           for n2, n3 in (("KNEE", "knee"), ("ANKLE", "ankle"), ("MP", "toe"))}
    log("  2D位置誤差RMS[mm]: " + str({k: round(v, 1) for k, v in err.items()}))

    summary = {"trial": tr.name, "source": t2.source, "kick": K, "tau0": tau[0], "tau1": tau[-1], "n": len(tau), "align": align,
               "lag_ms": lag * 1000, "phi_deg": np.degrees(phi) % 360, "psi_deg": np.degrees(psi) % 360,
               **{f"a_{k}_deg": v for k, v in zip(ALIGN_KEYS, a_deg)}, **{f"P_{k}_deg": v for k, v in zip(ALIGN_KEYS, p_deg)},
               **{f"before_{k}": v for k, v in before.items()}, **{f"after_{k}": v for k, v in after.items()},
               **{f"err2d_{k}_mm": v for k, v in err.items()}, **{f"L_{k}": v for k, v in c.L.items()}}
    out_dir = out_dir or OUT / tr.name
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "kinematics3d.csv", index=False, float_format="%.5f")
    plot(df, p2, tr, phi, out_dir / "check.png")
    return df, {"rot": r, "pos_G": pos, "psi": psi, "phi": phi, "L": c.L, "lag": lag, "tau": tau, "kick": K, "out_dir": out_dir}, summary


def plot(df: pd.DataFrame, p2: dict[str, np.ndarray], tr: Trial, phi: float, path: Path) -> None:
    K = tr.kick
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    fr = df["tau"]
    for a, n2, n3 in zip(ax[0], ("KNEE", "ANKLE", "MP"), ("knee", "ankle", "toe")):
        a.plot(fr, p2[n2][:, 0], "k", label="2D x"); a.plot(fr, df[f"{n3}{K}_x"], "r--", label="IMU-FK x")
        a.plot(fr, p2[n2][:, 1], "b", label="2D z"); a.plot(fr, df[f"{n3}{K}_z"], "c--", label="IMU-FK z")
        a.set_title(f"{n2} ({K})"); a.set_xlabel("time from impact [s]"); a.set_ylabel("m"); a.legend(fontsize=7)
    for a, jn in zip(ax[1], ("hip", "knee", "ankle")):
        for s, ls in (("L", "-"), ("R", "--")):
            for nm, col in (("flex", "r"), ("abd", "g"), ("rot", "b")):
                a.plot(fr, df[f"{jn}{s}_{nm}"], col + ls, label=f"{s} {nm}")
        a.set_title(f"{jn} angle [deg]"); a.set_xlabel("time from impact [s]"); a.legend(fontsize=6, ncol=2)
    fig.suptitle(f"{tr.name}: IMU(Xsens) + 2D video 3D lower-body (phi={np.degrees(phi) % 360:.0f} deg)")
    fig.tight_layout(); save_figure(fig, path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", default="kurogi_t1")
    ap.add_argument("--source", default="auto", choices=["auto", "pose2d", "kinovea"])
    ap.add_argument("--no-align", action="store_true", help="センサー間ヘディング補正を行わない")
    a = ap.parse_args()
    run(load_trials()[a.trial], a.source, align=not a.no_align)
