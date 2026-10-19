"""Xsens(IMU 7個)から、インパクト時の3D姿勢・スイング面などの「ボールの飛び方に効きそうな要因」を試行ごとに算出する。
側面カメラ1台の2Dでは見えない、横方向(蹴り足側)の傾き・体幹の側屈・スイング面の傾きを補う。 `python run.py factors` が自動で呼ぶ。

入力: data/xsens/<名前>/<日付>/<left|right>_shoot/xsens/taskN/{task,stop}/MT_*_<センサーID>.txt (task が無ければ taskN 直下)
      センサーIDと部位の対応は config/sensors.csv の kurogi_2023(2023〜2025の計測は同じ7個)
出力: session, trial_no, xs_* の列(kick_factors.csv に結合される)

方法(stop との差分は取らない。外部の同期表も不要):
 1. 姿勢 = Quat_q* があればそれ、無ければ Roll/Pitch/Yaw(ZYX)(kurogi で両者の一致を確認: 差 0.0°)。
    PacketCounter の抜け(フレーム欠落)は球面線形補間で100Hzに戻す。全体の欠落が15%超、または蹴りの区間(下腿ピークの前0.4秒〜後0.1秒)に
    欠落がある試行は除外する(yamada は欠落が3〜81%で、使えない)。
 2. インパクト = 蹴り脚の下腿の角速度最大の時刻(放物線補間) + 0.02秒。kurogi の動画由来のインパクトとの差(動画 - 下腿ピーク)が
    +0.01〜+0.03秒(n=6, SD 7ms)だったので、その平均を足している。角速度は姿勢の差分から求める(計測によって Gyr 列が無いため)。
    蹴りが検出できない試行(下腿ピーク < 1000deg/s、または蹴り脚/支持脚のピーク比 < 1.3)は除外する。
 3. 骨盤・大腿・下腿の「下向き」のセンサー座標 = 各試行で最も静止した0.5秒(直立)の重力方向の、被験者内の全試行の中央値。
    取り付けは被験者で異なる(kurogi・kataoka・kojima は -X、fujii は -Z が下向き)ため、被験者ごとに求める。取り付けのずれは一定のオフセットで、
    被験者内の相関には影響しない。
 4. 座標系 = 上(鉛直)・右(蹴り脚スイングの主回転軸)・前(= 上 × 右)。右は ω の主成分で決めるので方位に依存しない。
    角度は「蹴り足側が +」にそろえる(左足は左右反転して、右足と同じ意味)。
足部の向き(ピッチ・ヨー)は、足のセンサーの取り付けが被験者ごとに違い、ヘディングにも依存するため出さない。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R, Slerp

from .config import ROOT

FS = 100.0
Z = np.array([0.0, 0.0, 1.0])
SWING_S = 0.30  # インパクト前のスイング主軸を求める区間[s]
QUIET_S = 0.5  # 直立とみなす最も静止した区間[s]
SHANK_TO_IMPACT_S = 0.02  # 下腿ピーク → インパクト(kurogiの動画と比較して較正)
MIN_SHANK_DPS = 1000.0
MIN_KICK_RATIO = 1.3
MAX_MISSING = 0.15  # フレーム欠落率の上限(全体)。蹴りの区間(下腿ピークの前0.4秒〜後0.1秒)に欠落があれば別途除外
KICK_WINDOW_S = (0.4, 0.1)
PARTS = ("pelvis", "up_R", "leg_R", "foot_R", "up_L", "leg_L", "foot_L")
MOUNT_PARTS = ("pelvis", "up_R", "leg_R", "up_L", "leg_L")  # 下向き軸を求める部位(足は取り付けが一定でないので使わない)


def sensor_ids(config_dir: Path | None = None) -> dict[str, str]:
    p = (config_dir or ROOT / "config") / "sensors.csv"
    df = pd.read_csv(p, dtype=str)
    return dict(zip(*(df[df["sensor_set"] == "kurogi_2023"][c] for c in ("part", "id"))))


def xsens_dir(name: str, side: str, data_root: Path | None = None) -> Path | None:
    base = (data_root or ROOT / "data" / "xsens") / name
    hits = sorted(base.glob(f"*/{side}_shoot/xsens")) if base.is_dir() else []
    return hits[-1] if hits else None


def _load(d: Path, sid: str) -> tuple[R, float, int] | None:
    """(100Hzに揃えた姿勢, フレーム欠落率, 最大の連続欠落)。姿勢は Quat_q* があればそれを、無ければ Roll/Pitch/Yaw(ZYX)から作る。"""
    f = next(iter(sorted(d.glob(f"*{sid}.txt"))), None)
    if f is None:
        return None
    df = pd.read_csv(f, comment="/").dropna(axis=1, how="all")
    q = [c for c in ("Quat_q1", "Quat_q2", "Quat_q3", "Quat_q0") if c in df]
    if len(q) == 4:
        rot = R.from_quat(df[q].to_numpy())
    else:
        rot = R.from_euler("ZYX", df[["Yaw", "Pitch", "Roll"]].to_numpy(), degrees=True)
    pc = df["PacketCounter"].to_numpy(float)
    idx = np.unwrap(pc * 2 * np.pi / 65536) * 65536 / (2 * np.pi)  # カウンタは65535で一周する
    idx = np.round(idx - idx[0])
    keep = np.r_[True, np.diff(idx) > 0]  # 重複・逆戻りは捨てる
    rot, idx = rot[keep], idx[keep]
    span = int(idx[-1]) + 1
    missing = max(0.0, 1 - len(idx) / span)
    filled = np.ones(span, bool)  # True = 欠落を補間したフレーム
    filled[idx.astype(int)] = False
    if missing > 0:
        rot = Slerp(idx, rot)(np.arange(span))
    return rot, float(missing), filled


def _omega_world(rot: R) -> np.ndarray:
    """姿勢の差分から求めた世界系の角速度 [rad/s](100Hz。計測によって Gyr 列が無いので全員これでそろえる)。"""
    d = (rot[1:] * rot[:-1].inv()).as_rotvec() * FS
    return np.vstack([d, d[-1:]])


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _deg(x: float) -> float:
    return float(np.degrees(x))


def _peak(y: np.ndarray, i: int) -> float:
    """離散ピーク位置 i の放物線補間(小数のサンプル位置)。"""
    y0, y1, y2 = y[i - 1:i + 2]
    den = y0 - 2 * y1 + y2
    return i + (float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5)) if den < 0 else 0.0)


def load_trial(task_dir: Path, ids: dict[str, str]) -> tuple[dict[str, R], np.ndarray]:
    """(部位ごとの姿勢, いずれかのセンサーが欠落を補間したフレームのマスク)"""
    data = {p: _load(task_dir, ids[p]) for p in PARTS}
    if any(v is None for v in data.values()):
        raise FileNotFoundError(f"センサーが足りません: {task_dir}")
    miss = max(v[1] for v in data.values())
    if miss > MAX_MISSING:
        raise ValueError(f"フレーム欠落が多い(最大 {miss * 100:.0f}%)")
    n = min(len(v[0]) for v in data.values())
    filled = np.any([np.pad(v[2], (0, max(0, n - len(v[2]))))[:n] for v in data.values()], axis=0)
    return {p: v[0][:n] for p, v in data.items()}, filled


def quiet_down(rot: dict[str, R]) -> dict[str, np.ndarray]:
    """最も静止した0.5秒の、センサー座標での鉛直下向き(直立のときセグメントの下向き軸に近い)。"""
    w = int(QUIET_S * FS)
    tot = sum(np.linalg.norm(_omega_world(rot[p]), axis=1) for p in MOUNT_PARTS)
    s0 = int(np.argmin(np.convolve(tot, np.ones(w) / w, mode="valid")))
    return {p: rot[p][s0:s0 + w].mean().inv().apply(-Z) for p in MOUNT_PARTS}


def trial_factors(rot: dict[str, R], filled: np.ndarray, kick_side: str, mount: dict[str, np.ndarray]) -> dict:
    sup = "L" if kick_side == "right" else "R"
    K = "R" if kick_side == "right" else "L"
    n = len(rot["pelvis"])
    om = {p: _omega_world(rot[p]) for p in PARTS}
    gn = {p: np.linalg.norm(o, axis=1) for p, o in om.items()}

    # 2. インパクト
    ip = int(np.argmax(gn[f"leg_{K}"]))
    peak_dps = _deg(gn[f"leg_{K}"][ip])
    ratio = float(gn[f"leg_{K}"].max() / max(gn[f"leg_{sup}"].max(), 1e-9))
    if peak_dps < MIN_SHANK_DPS or ratio < MIN_KICK_RATIO or ip < 2 or ip > n - 3:
        raise ValueError(f"蹴りを検出できません(下腿ピーク {peak_dps:.0f} deg/s、蹴り脚/支持脚 {ratio:.1f})")
    if filled[max(0, ip - int(KICK_WINDOW_S[0] * FS)):ip + int(KICK_WINDOW_S[1] * FS)].any():
        raise ValueError("蹴りの区間でフレームが欠落している")
    tp = _peak(gn[f"leg_{K}"], ip)
    ti = min(tp + SHANK_TO_IMPACT_S * FS, n - 1.0)  # インパクトのサンプル位置
    i1 = int(round(ti))
    at = {q: Slerp(np.arange(n), rot[q])([ti])[0] for q in PARTS}  # インパクト時の姿勢(球面線形補間)

    # 4. 座標系(上, 右, 前)
    a, b = max(0, i1 - int(SWING_S * FS)), i1 + 1
    w_world = om[f"leg_{K}"][a:b]
    wts = np.linalg.norm(w_world, axis=1)
    cov = (w_world * wts[:, None]).T @ w_world
    val, vec = np.linalg.eigh(cov)
    r = vec[:, np.argmax(val)]
    if (w_world * wts[:, None]).sum(axis=0) @ r < 0:
        r = -r  # ωの向き = 被験者の右(前方への振り出しで)
    rh = _unit(r - (r @ Z) * Z)
    f = np.cross(Z, rh)
    lk = rh if kick_side == "right" else -rh  # 蹴り足側の横方向(水平)
    lk3 = r if kick_side == "right" else -r  # 同じ向きのスイング主軸(傾きを含む)

    def down(p: str) -> np.ndarray:
        return at[p].apply(mount[p])

    def tilt(v: np.ndarray) -> tuple[float, float]:
        """下向き軸の(前方への振り出し角, 蹴り足側への傾き) [deg]"""
        return _deg(np.arctan2(v @ f, -(v @ Z))), _deg(np.arcsin(np.clip(v @ lk, -1, 1)))

    out: dict = {}
    for tag, p in (("kick_thigh", f"up_{K}"), ("kick_shank", f"leg_{K}"), ("support_thigh", f"up_{sup}"), ("support_shank", f"leg_{sup}")):
        out[f"xs_{tag}_fwd_deg"], out[f"xs_{tag}_lat_deg"] = tilt(down(p))
    up = -down("pelvis")
    out["xs_pelvis_lean_fwd_deg"] = _deg(np.arctan2(up @ f, up @ Z))
    out["xs_pelvis_lean_kickside_deg"] = _deg(np.arctan2(up @ lk, up @ Z))
    out["xs_swing_axis_tilt_deg"] = _deg(np.arcsin(np.clip(lk3 @ Z, -1, 1)))  # スイング面の傾き(蹴り足側が上がる+)

    # 運動連鎖のタイミング・速さ
    t0 = max(0, ip - 30)
    out["xs_impact_t_s"] = ti / FS
    out["xs_shank_gyro_peak_dps"] = peak_dps
    out["xs_thigh_gyro_peak_dps"] = _deg(gn[f"up_{K}"][t0:ip + 6].max())
    out["xs_thigh_peak_to_shank_peak_ms"] = (_peak(gn[f"up_{K}"], t0 + int(np.argmax(gn[f"up_{K}"][t0:ip + 6]))) - tp) / FS * 1000  # <0=大腿が先
    out["xs_kickleg_ratio"] = ratio  # 蹴り脚のセンサー対応の確認値(>1.3 で採用)
    return out


def _exclusions(session: str) -> list[tuple[int | None, str, str]]:
    """config/xsens_exclude.csv の除外 [(trial_no(None=全試行), scope, reason)]。scope: all(Xsensを使わない) / posture(姿勢角・3Dは使わない)。"""
    p = ROOT / "config" / "xsens_exclude.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p, comment="#", dtype=str, keep_default_na=False)
    df = df[df["session"] == session]
    return [((int(r.trial_no) if r.trial_no.strip() else None), (getattr(r, "scope", "") or "all").strip(), r.reason) for r in df.itertuples()]


def _excluded(session: str) -> dict[int | None, str]:
    """Xsens を一切使わない試行(scope=all)。キーが None ならその被験者の全試行。"""
    return {k: why for k, sc, why in _exclusions(session) if sc == "all"}


def posture_off(session: str) -> dict[int | None, str]:
    """姿勢角・3Dを使わない試行(scope=posture)。キーが None ならその被験者の全試行。"""
    return {k: why for k, sc, why in _exclusions(session) if sc == "posture"}


def _is_posture_key(k: str) -> bool:
    return k.startswith("xs_") and (k.endswith(("_fwd_deg", "_lat_deg")) or k.startswith("xs_pelvis_lean") or k == "xs_swing_axis_tilt_deg")


def xsens_factors(session: str, kick_side: str, data_root: Path | None = None, log=print) -> pd.DataFrame:
    d = xsens_dir(session, kick_side, data_root)
    if d is None:
        return pd.DataFrame()
    ids = sensor_ids()
    ex = _excluded(session)
    if None in ex:
        log(f"[{session}] Xsens は除外設定(config/xsens_exclude.csv): {ex[None]}")
        return pd.DataFrame()
    loaded: dict[int, tuple[dict[str, R], np.ndarray]] = {}
    for t in sorted(d.glob("task*"), key=lambda p: int(re.sub(r"\D", "", p.name) or 0)):
        no = int(re.sub(r"\D", "", t.name) or 0)
        if no in ex:
            continue
        try:
            loaded[no] = load_trial(t / "task" if (t / "task").is_dir() else t, ids)
        except Exception as e:  # 1試行の失敗で全体を止めない
            log(f"[スキップ] Xsens {session}/{t.name}: {type(e).__name__}: {e}")
    if not loaded:
        return pd.DataFrame()
    qd = [quiet_down(r) for r, _ in loaded.values()]
    mount = {p: _unit(np.median([q[p] for q in qd], axis=0)) for p in MOUNT_PARTS}  # 被験者ごとの下向き軸(センサー座標)
    rows = []
    po = posture_off(session)
    if po:
        log(f"[{session}] Xsens は姿勢角を使わない設定(scope=posture): " + "; ".join(po.values()))
    for no, (rot, filled) in loaded.items():
        try:
            f = trial_factors(rot, filled, kick_side, mount)
        except Exception as e:
            log(f"[スキップ] Xsens {session}/task{no}: {type(e).__name__}: {e}")
            continue
        if None in po or no in po:  # 直立基準に依存する値は欠損にする(時間差・角速度ピークは残す)
            f = {k: (np.nan if _is_posture_key(k) else v) for k, v in f.items()}
        rows.append(dict(session=session, trial_no=no, **f))
    return pd.DataFrame(rows)
