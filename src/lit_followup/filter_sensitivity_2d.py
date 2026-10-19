"""ローパスのカットオフを変えたとき、インパクト付近の指標(2D高速度映像)がどれだけ動くか。

背景: Nunome ら(2006; ISBS基調講演)は、10Hz程度で平滑化すると、インパクト直前に足部・下腿の偽の減速が出ると報告している。
このリポジトリの既定は15Hz(config/pipeline.toml)。raw_px.csv(未フィルタの検出点)から、カットオフを変えて
process() → trial_factors() をやり直し、足先速度の仰角・つま先速度ピークの時刻・ボール出射仰角との差・大腿-下腿のピーク時刻差が
どれだけ変わるかを見る。既存の output/<session>/<trial>/ は書き換えず、出力は output/lit_followup/ のみ。
raw_px.csv は動画と同じ共用PCにしか無い試行が多い(この PC には kurogi_left_task1 だけ)。共用PCで実行すれば全試行を処理できる。
高カットオフは検出点のノイズを通すので、`speed_roughness` と合わせて判断する。

実行: PYTHONPATH=src python -m lit_followup.filter_sensitivity_2d [--session S] [--trial T]
出力: output/lit_followup/01_filter_sensitivity_2d/{filter_sensitivity_2d.csv, filter_sensitivity_2d.png(+_ja)}
"""
from __future__ import annotations

import argparse
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pose2d.config import ROOT, ProcessCfg
from pose2d.figio import save_figure
from pose2d.kickfactors import trial_factors
from pose2d.mechanics2d import kick_sign
from pose2d.process import process
from pose2d.sessions import load_session

from ._common import OUT_DIR

warnings.filterwarnings("ignore", category=RuntimeWarning)
SUBDIR = "01_filter_sensitivity_2d"
CUTOFFS_HZ = (8, 10, 15, 25, 40, 60, 100, 150)
REF_HZ = 15
PEAK_WINDOW_S = (-0.03, 0.01)    # つま先速度ピークを探す区間(インパクト基準)
CHAIN_WINDOW_S = (-0.35, 0.0)    # 大腿・下腿の角速度ピークを探す区間
MIN_CONF = 0.3


def impacts() -> dict[tuple[str, str], int]:
    df = pd.read_csv(ROOT / "config" / "impacts.csv", encoding="utf-8-sig")
    return {(str(r.session), str(r.trial)): int(r.impact_frame) for r in df.itertuples()}


def find_impact(tbl: dict[tuple[str, str], int], session: str, trial: str) -> int | None:
    """kurogi_left_task2 と task2 を同じ試行として扱う(cli.find_impact と同じ規則)。"""
    if (session, trial) in tbl:
        return tbl[(session, trial)]
    key = re.sub(r"^.*?(task\d+)$", r"\1", trial)
    for (s, t), f in tbl.items():
        if s == session and re.sub(r"^.*?(task\d+)$", r"\1", t) == key:
            return f
    return None


def available() -> list[tuple[str, str]]:
    return sorted((p.parent.parent.name, p.parent.name) for p in (ROOT / "output").glob("*/*/raw_px.csv"))


def analyze(session: str, trial: str) -> tuple[list[dict], dict]:
    sess = load_session(session)
    cal, fps = sess.load_calibration(), float(sess.fps)
    impact = find_impact(impacts(), session, trial)
    if impact is None:
        raise RuntimeError("impacts.csv にインパクトが登録されていません")
    raw = pd.read_csv(ROOT / "output" / session / trial / "raw_px.csv")
    side = sess.kick_side[0].upper()
    rows, traces = [], {}
    for fc in CUTOFFS_HZ:
        kin, info = process(raw, cal, fps, ProcessCfg(cutoff_hz=float(fc)), MIN_CONF, sess.kick_side, impact)
        f = trial_factors(kin, info)
        t = kin["t_rel_impact_s"].to_numpy()
        sp = kin[f"{sess.kick_side}_big_toe_speed"].to_numpy()
        w = (t >= PEAK_WINDOW_S[0]) & (t <= PEAK_WINDOW_S[1])
        ipk = np.flatnonzero(w)[int(np.nanargmax(sp[w]))]
        c = (t >= CHAIN_WINDOW_S[0]) & (t <= CHAIN_WINDOW_S[1])
        om_t = np.abs(kin[f"thigh_{side}_omega_dps"].to_numpy())
        om_s = np.abs(kin[f"shank_{side}_omega_dps"].to_numpy())
        i_t, i_s = np.flatnonzero(c)[int(np.nanargmax(om_t[c]))], np.flatnonzero(c)[int(np.nanargmax(om_s[c]))]
        r0 = np.flatnonzero((t >= -0.1) & (t <= -0.05))
        sgn = kick_sign(kin, sess.kick_side)
        elev = np.degrees(np.arctan2(kin[f"{sess.kick_side}_big_toe_vy"].to_numpy(), sgn * kin[f"{sess.kick_side}_big_toe_vx"].to_numpy()))
        at = lambda ms: float(np.nanmean(elev[(t >= (ms - 5) / 1000) & (t <= ms / 1000)]))  # noqa: E731  直前5msの平均
        rows.append(dict(session=session, trial=trial, cutoff_hz=fc, fps=fps, toe_vel_elev_deg=f["toe_vel_elev_deg"],
                         toe_elev_m20ms_deg=at(-20), toe_elev_m10ms_deg=at(-10), toe_elev_m5ms_deg=at(-5),
                         toe_vel_mps=f["toe_vel_mps"], launch_minus_toevel_deg=f["launch_minus_toevel_deg"],
                         ball_launch_elev_deg=f["ball_launch_elev_deg"], toe_peak_time_ms=1000 * t[ipk],
                         toe_peak_speed_mps=sp[ipk], toe_speed_at_impact_over_peak=float(sp[np.nanargmin(np.abs(t))] / sp[ipk]),
                         thigh_peak_ms=1000 * t[i_t], shank_peak_ms=1000 * t[i_s], shank_minus_thigh_peak_ms=1000 * (t[i_s] - t[i_t]),
                         speed_roughness=float(np.sqrt(np.nanmean(np.diff(sp[r0], 2) ** 2)))))
        traces[fc] = (1000 * t, sp, elev)
    return rows, traces


def build_figure(df: pd.DataFrame, traces: dict) -> plt.Figure:
    keys = sorted(traces)
    fig, axes = plt.subplots(len(keys), 6, figsize=(20, 3.4 * len(keys)), constrained_layout=True, squeeze=False)
    cmap = plt.get_cmap("viridis")
    for i, (session, trial) in enumerate(keys):
        d = df[(df.session == session) & (df.trial == trial)]
        for ax, idx, ylabel, lim in ((axes[i, 0], 1, "Toe speed (m/s)", (-80, 40)), (axes[i, 1], 2, "Toe-velocity elevation at impact (deg)", (-50, 5))):
            for k, fc in enumerate(CUTOFFS_HZ):
                tr_ = traces[(session, trial)][fc]
                t, y = tr_[0], tr_[idx]
                sel = (t >= lim[0]) & (t <= lim[1])
                ax.plot(t[sel], y[sel], color="k" if fc == REF_HZ else cmap(k / (len(CUTOFFS_HZ) - 1)), lw=2.2 if fc == REF_HZ else 0.9, label=f"{fc} Hz")
            ax.axvline(0, color="#c33", ls="--", lw=0.8)
            ax.set_xlabel("Time from impact (ms)", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(alpha=0.25)
        axes[i, 0].set_title(f"{session}/{trial}", fontsize=8)
        axes[i, 0].legend(fontsize=6, ncol=2)
        for ax, (col, label) in zip(axes[i, 2:], [("toe_vel_elev_deg", "Toe-velocity elevation at impact (deg)"),
                                                  ("launch_minus_toevel_deg", "Ball elevation - toe-velocity elevation (deg)"),
                                                  ("toe_peak_time_ms", "Toe-speed peak time vs impact (ms)"),
                                                  ("shank_minus_thigh_peak_ms", "Shank-peak minus thigh-peak time (ms)")]):
            ax.plot(d.cutoff_hz, d[col], marker="o", ms=3)
            ax.axvline(REF_HZ, color="k", ls=":", lw=0.8)
            ax.set_xscale("log")
            ax.set_xlabel("Low-pass cutoff (Hz)", fontsize=8)
            ax.set_ylabel(label, fontsize=8)
            ax.grid(alpha=0.25)
        for ax in axes[i]:
            ax.tick_params(labelsize=7)
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--session")
    ap.add_argument("--trial")
    a = ap.parse_args()
    todo = [(s, t) for s, t in available() if (not a.session or s == a.session) and (not a.trial or t == a.trial)]
    if not todo:
        raise SystemExit("raw_px.csv のある試行がありません(動画のある共用PCで実行するか、run.py run で raw_px.csv を作る)")
    out = OUT_DIR / SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    rows, traces = [], {}
    for s, t in todo:
        print(f"{s}/{t} ...", flush=True)
        try:
            r, traces[(s, t)] = analyze(s, t)
            rows += r
        except Exception as e:  # 1試行の失敗で止めない
            print(f"  [スキップ] {e}")
    df = pd.DataFrame(rows)
    df.round(4).to_csv(out / "filter_sensitivity_2d.csv", index=False)
    print(df[["trial", "cutoff_hz", "toe_vel_elev_deg", "launch_minus_toevel_deg", "toe_peak_time_ms", "toe_speed_at_impact_over_peak",
              "shank_minus_thigh_peak_ms", "speed_roughness"]].round(3).to_string(index=False))
    fig = build_figure(df, traces)
    save_figure(fig, out / "filter_sensitivity_2d.png", dpi=130)
    plt.close(fig)
