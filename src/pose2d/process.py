"""生キーポイント(px) → フィルタ済み実寸(m)運動学。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from .calib import Calibration
from .config import ProcessCfg
from .keypoints import OUTPUT_KPS, SEGMENTS


# ---------- 前処理 ----------
def _limit_fill(s: pd.Series, max_gap: int) -> pd.Series:
    """max_gap以下の欠損のみ線形補間。それ以上の欠損はNaNのまま。"""
    filled = s.interpolate(method="linear", limit_area="inside")
    isna = s.isna()
    grp = (isna != isna.shift()).cumsum()
    run = isna.groupby(grp).transform("sum")
    return filled.where(~(isna & (run > max_gap)), np.nan)


def _despike(x: pd.Series, k: float) -> pd.Series:
    """前後に戻る1点だけのジャンプ(フレーム間変位が中央値+k*MAD超)を欠損化。"""
    d = x.diff().abs()
    med = d.median()
    mad = (d - med).abs().median() * 1.4826
    if not np.isfinite(mad) or mad == 0:
        return x
    thr = med + k * mad
    return x.mask((d > thr) & (d.shift(-1) > thr))


def _lowpass(x: np.ndarray, fc: float, fs: float, order: int, pad: int) -> np.ndarray:
    b, a = butter(order, fc / (fs / 2))
    return filtfilt(b, a, x, padlen=min(pad, len(x) - 1))


def _filter_valid(s: pd.Series, fc: float, fs: float, cfg: ProcessCfg) -> pd.Series:
    """長い欠損を跨がないよう、連続有効区間ごとにフィルタ。短すぎる区間はNaN。"""
    out = pd.Series(np.nan, index=s.index)
    valid = s.notna()
    grp = (valid != valid.shift()).cumsum()
    min_len = max(3 * cfg.filter_order, 12)
    for _, seg in s[valid].groupby(grp[valid]):
        if len(seg) >= min_len:
            out.loc[seg.index] = _lowpass(seg.to_numpy(), fc, fs, cfg.filter_order, cfg.pad_frames)
    return out


def winter_cutoff(series: list[np.ndarray], fs: float, order: int = 4) -> float:
    """Winter残差解析。RMS残差-fc曲線の高周波域を直線近似し、切片と交わるfcを返す。"""
    fcs = np.arange(2.0, min(0.4 * fs, 100.0), 1.0)
    res = np.zeros_like(fcs)
    for x in series:
        for j, fc in enumerate(fcs):
            res[j] += np.sqrt(np.mean((x - _lowpass(x, fc, fs, order, 50)) ** 2))
    res /= len(series)
    n = max(4, len(fcs) // 4)
    _, icpt = np.polyfit(fcs[-n:], res[-n:], 1)
    below = np.flatnonzero(res <= icpt)
    return float(fcs[below[0]]) if len(below) else float(fcs[-1])


# ---------- イベント ----------
def detect_impact(ball_m: np.ndarray, frames: np.ndarray, thresh_m: float = 0.03) -> int | None:
    """ボールが初期位置から thresh_m 動く直前のフレームをインパクトとする(要目視確認)。"""
    ok = np.isfinite(ball_m).all(1)
    if ok.sum() < 20:
        return None
    ref = np.nanmedian(ball_m[np.flatnonzero(ok)[:15]], axis=0)
    dist = np.linalg.norm(ball_m - ref, axis=1)
    moved = np.flatnonzero(ok & (dist > thresh_m))
    if not len(moved):
        return None
    i = moved[0]
    while i > 0 and not (ok[i - 1] and dist[i - 1] < thresh_m / 3):
        i -= 1
    return int(frames[max(i - 1, 0)])


def ball_speed_after(ball_m: np.ndarray, t: np.ndarray, t_impact: float, window: tuple[float, float] = (0.01, 0.02)
                     ) -> float | None:
    """インパクト後 window(秒)のボール位置を直線近似した速さ [m/s] (SI2025: 0.01〜0.02 s)。"""
    sel = (t - t_impact >= window[0] - 1e-9) & (t - t_impact <= window[1] + 1e-9)
    p, tt = ball_m[sel], t[sel]
    ok = np.isfinite(p).all(1)
    if ok.sum() < 3:
        return None
    return float(np.hypot(*[np.polyfit(tt[ok], p[ok, i], 1)[0] for i in range(2)]))


def _unwrap_nan(a: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan)
    ok = np.isfinite(a)
    out[ok] = np.unwrap(a[ok])
    return out


def _wrap180(a):
    return (a + 180) % 360 - 180


# ---------- メイン ----------
def process(raw: pd.DataFrame, cal: Calibration, fps: float, cfg: ProcessCfg,
            min_conf: float = 0.3, kick_side: str = "right", impact_frame: int | None = None,
            ) -> tuple[pd.DataFrame, dict]:
    """戻り値: (時系列DataFrame[m, s, deg], サマリdict)。"""
    df = raw.copy()
    kps = [k for k in OUTPUT_KPS if f"{k}_x" in df and df[f"{k}_x"].notna().any()]

    # 1) 信頼度マスク・スパイク除去・短い欠損の補間 (px)
    for k in kps:
        low = df[f"{k}_c"] < min_conf
        for ax in "xy":
            s = df[f"{k}_{ax}"].mask(low)
            df[f"{k}_{ax}"] = _limit_fill(_despike(s, cfg.spike_mad), cfg.max_gap_frames)

    # 2) 遮断周波数
    if str(cfg.cutoff_hz).lower() == "auto":
        lim = [f"{kick_side}_{j}" for j in ("hip", "knee", "ankle", "big_toe", "heel") if f"{kick_side}_{j}" in kps]
        ser = [df[f"{k}_{a}"].interpolate(limit_area="inside").dropna().to_numpy() for k in lim for a in "xy"]
        ser = [s for s in ser if len(s) > 60]
        fc = winter_cutoff(ser, fps, cfg.filter_order) if ser else 15.0
    else:
        fc = float(cfg.cutoff_hz)
    fc = min(fc, 0.4 * fps)
    info: dict = {"fps": fps, "n_frames": len(df), "mm_per_px_x": cal.mm_per_px_x,
                  "mm_per_px_y": cal.mm_per_px_y, "kick_side": kick_side, "cutoff_hz": fc}

    # 3) フィルタ → m変換
    out = pd.DataFrame({"frame": df["frame"], "time_s": df["frame"] / fps})
    for k in kps:
        xf = _filter_valid(df[f"{k}_x"], fc, fps, cfg)
        yf = _filter_valid(df[f"{k}_y"], fc, fps, cfg)
        m = cal.px_to_m(np.column_stack([xf, yf]))
        out[f"{k}_x"], out[f"{k}_y"] = m[:, 0], m[:, 1]

    # 4) ボール(ローパスせず短い欠損の補間のみ)とインパクト
    if "ball_x" in df and df["ball_x"].notna().any():
        b = np.column_stack([_limit_fill(df["ball_x"], 10), _limit_fill(df["ball_y"], 10)])
        m = cal.px_to_m(b)
        out["ball_x"], out["ball_y"] = m[:, 0], m[:, 1]
        if impact_frame is None:
            impact_frame = detect_impact(m, out["frame"].to_numpy())
            info["impact_source"] = "auto(ball)"
        if impact_frame is not None:
            info["ball_speed_mps"] = ball_speed_after(m, out["time_s"].to_numpy(), impact_frame / fps)
    if impact_frame is not None:
        info["impact_frame"] = int(impact_frame)

    # 5) 微分量(フィルタ済み位置の中心差分)
    dt = 1.0 / fps
    for k in kps:
        for a in "xy":
            out[f"{k}_v{a}"] = np.gradient(out[f"{k}_{a}"].to_numpy(), dt)
        out[f"{k}_speed"] = np.hypot(out[f"{k}_vx"], out[f"{k}_vy"])
    if "ball_x" in out:
        out["ball_speed"] = np.hypot(np.gradient(out["ball_x"].to_numpy(), dt),
                                     np.gradient(out["ball_y"].to_numpy(), dt))

    for name, prox, dist in SEGMENTS:
        if f"{prox}_x" in out and f"{dist}_x" in out:
            ang = np.arctan2(out[f"{dist}_y"] - out[f"{prox}_y"], out[f"{dist}_x"] - out[f"{prox}_x"])
            out[f"{name}_angle_deg"] = np.degrees(_unwrap_nan(ang.to_numpy()))
            out[f"{name}_omega_dps"] = np.gradient(out[f"{name}_angle_deg"].to_numpy(), dt)
    for s in "RL":
        if f"thigh_{s}_angle_deg" in out and f"shank_{s}_angle_deg" in out:
            # 膝屈曲: 大腿と下腿が一直線 = 0°
            out[f"knee_flexion_{s}_deg"] = _wrap180(out[f"thigh_{s}_angle_deg"] - out[f"shank_{s}_angle_deg"]).abs()

    if impact_frame is not None:
        out.insert(2, "t_rel_impact_s", (out["frame"] - impact_frame) * dt)
        r = out.index[out["frame"] == impact_frame]
        s = kick_side[0].upper()
        if len(r):
            for col, key in ((f"{kick_side}_big_toe_speed", "foot_tip_speed_at_impact_mps"),
                             (f"{kick_side}_ankle_speed", "ankle_speed_at_impact_mps"),
                             (f"shank_{s}_omega_dps", "shank_omega_at_impact_dps")):
                if col in out and np.isfinite(out.loc[r[0], col]):
                    info[key] = float(out.loc[r[0], col])
    return out, info
