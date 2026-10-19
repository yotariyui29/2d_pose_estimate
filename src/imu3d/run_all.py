"""全試行（または指定）を一括処理: 3D運動学 → output/imu3d/<trial>/ 、OpenSim出力、output/imu3d/summary.csv。
2Dデータが無い試行・解析区間が短い試行はスキップして一覧に出す。1試行が失敗しても続行する。

Usage:
  python run.py imu3d                            # config/trials.csv の全試行
  python run.py imu3d --trials kurogi_t1,kurogi_t3 [--source pose2d] [--no-opensim] [--no-align]
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import pandas as pd

from .imu_video_3d import OUT, run
from .trials import load_paths, load_trials


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", default="all", help="all または カンマ区切りの trial 名")
    ap.add_argument("--source", default="auto", choices=["auto", "pose2d", "kinovea"])
    ap.add_argument("--no-align", action="store_true", help="センサー間ヘディング補正を行わない")
    ap.add_argument("--no-opensim", action="store_true", help="OpenSim出力(.osim/.mot/.trc)を作らない")
    ap.add_argument("--init", action="store_true", help="data/xsens の被験者の試行を config/trials.csv に追加して終了")
    a = ap.parse_args()
    if a.init:
        from .init_trials import init_trials

        init_trials()
        return

    trials = load_trials()
    names = list(trials) if a.trials == "all" else [s.strip() for s in a.trials.split(",")]
    unknown = [n for n in names if n not in trials]
    if unknown:
        sys.exit(f"config/trials.csv に無い試行: {unknown}（あるのは {list(trials)}）")
    export = None
    if not a.no_opensim:
        try:
            from .export_opensim import export
        except ImportError as e:
            print(f"opensim が使えないため OpenSim出力をスキップ: {e}")

    rows, skipped, failed = [], [], []
    for n in names:
        tr = trials[n]
        try:
            res = run(tr, a.source, align=not a.no_align)
            if res is None:
                skipped.append(n)
                continue
            rows.append(res[2])
            if export:
                export(tr, a.source, not a.no_align, res=res)
        except Exception:
            traceback.print_exc()
            failed.append(n)
    if rows:
        path = OUT / "summary.csv"
        new = pd.DataFrame(rows)
        if path.exists() and a.trials != "all":  # 一部だけ再実行したときは既存行を保持
            old = pd.read_csv(path)
            new = pd.concat([old[~old["trial"].isin(new["trial"])], new], ignore_index=True)
        OUT.mkdir(parents=True, exist_ok=True)
        new.to_csv(path, index=False, float_format="%.4f")
        print(f"summary: {path}")
    print(f"完了 {len(rows)} / スキップ(2Dなし・区間短) {skipped} / 失敗 {failed}")


if __name__ == "__main__":
    main()
