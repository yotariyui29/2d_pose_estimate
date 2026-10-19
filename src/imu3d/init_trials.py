"""data/xsens にある被験者の試行を config/trials.csv に追加する。  python run.py imu3d --init

- インパクト時刻 = Xsens の蹴り脚の下腿ピーク + 0.02秒(pose2d.xsensfactors)。start_s = それ - 0.9 [秒、記録の先頭=0]。
  動画のインパクトとは±数十msずれるが、3Dの当てはめが lag(±0.1秒)を推定するので問題ない。
- 対象は sessions.csv の被験者のうち Xsens(data/xsens/<名前>/)があり、config/xsens_exclude.csv で除外されていないもの。
  蹴りが検出できない試行・フレーム欠落が多い試行は追加しない。既にある試行(trial名)は触らない。
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from pose2d.sessions import list_sessions, load_session
from pose2d.xsensfactors import _excluded, posture_off, xsens_dir, xsens_factors

from .trials import CONFIG, ROOT

COLS = ["trial", "subject", "kick", "sensor_set", "xsens_subdir", "start_s", "pose2d_trial", "kinovea_csv", "kinovea_fps",
        "kinovea_flip_x", "note", "xsens_path"]


def new_rows(existing: set[str], log=print) -> list[dict]:
    rows = []
    for name in list_sessions():
        side = load_session(name).kick_side
        d = xsens_dir(name, side)
        po = posture_off(name)  # 3Dは直立基準の姿勢が要るので、姿勢角を使わない被験者・試行は対象外
        if d is None or None in _excluded(name) or None in po:
            continue
        xf = xsens_factors(name, side, log=log)
        for r in xf.itertuples():
            if int(r.trial_no) in po:
                continue
            trial = f"{name}_t{int(r.trial_no)}"
            if trial in existing:
                continue
            rel = d.relative_to(ROOT) / f"task{int(r.trial_no)}"
            rows.append(dict(trial=trial, subject=name, kick=side[0].upper(), sensor_set="kurogi_2023", xsens_subdir=f"task{int(r.trial_no)}",
                             start_s=f"{r.xs_impact_t_s - 0.9:.4f}", pose2d_trial=f"{name}_{side}_T{int(r.trial_no)}", kinovea_csv="",
                             kinovea_fps="1000", kinovea_flip_x="0", note="自動追加(Xsensのインパクト検出)", xsens_path=rel.as_posix()))
    return rows


def init_trials(log=print) -> int:
    path = CONFIG / "trials.csv"
    raw = path.read_bytes().decode("utf-8-sig")
    existing = {r["trial"] for r in csv.DictReader(raw.splitlines())}
    rows = new_rows(existing, log)
    if not rows:
        log("追加する試行はありません")
        return 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, lineterminator="\n")
        for r in rows:
            w.writerow(r)
    log(f"{len(rows)} 試行を {path} に追加: " + ", ".join(sorted({re.sub(r'_t\d+$', '', r['trial']) for r in rows})))
    return len(rows)
