"""共用PCの解析結果を、git で別のPCへ渡すためにまとめる(export)と、受け取った側で output/ に戻す(import-results)。

  共用PC:  python run.py export [--with-kinematics]   → results/ に集める → git add results; git commit; git push
  別のPC:  git pull → python run.py import-results     → results/ の内容を output/ に戻す(factors・contact・chain を再実行できる)

results/ に入れるもの(動画・raw_px.csv・重畳動画は入れない):
  analysis/        output/analysis 全体(表・図。英語版と _ja の日本語版)
  imu3d/           output/imu3d の summary.csv、各試行の check.png・check_ja.png・kinematics3d.csv、opensim/ の .osim/.mot/.trc(Geometry は除く)
  trials/<被験者>/<試行>/  summary.json、plot.png、plot_ja.png(--with-kinematics があれば kinematics.csv も)
  summary_all.csv, <被験者>/summary.csv
  MANIFEST.txt     実行日時・コミット・ファイル数・容量
"""
from __future__ import annotations

import datetime as dt
import shutil
import subprocess
from pathlib import Path

from .config import ROOT

TRIAL_FILES = ("summary.json", "plot.png", "plot_ja.png")


def _copy(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return src.stat().st_size


def run_export(with_kinematics: bool = False, out: Path | None = None, root: Path | None = None, log=print) -> Path:
    root = root or ROOT / "output"
    out = out or ROOT / "results"
    if out.exists():  # 前回の内容は消して作り直す(古い試行が残らないように)
        shutil.rmtree(out)
    n, size = 0, 0

    def cp(src: Path, dst: Path) -> None:
        nonlocal n, size
        size += _copy(src, dst)
        n += 1

    if (root / "analysis").is_dir():
        for f in (root / "analysis").rglob("*"):
            if f.is_file():
                cp(f, out / "analysis" / f.relative_to(root / "analysis"))
    if (root / "imu3d").is_dir():
        for f in (root / "imu3d").rglob("*"):
            if f.is_file() and (f.name in ("summary.csv", "kinematics3d.csv") or f.name.startswith(("check", "animation"))
                                or (f.parent.name == "opensim" and f.suffix in (".mot", ".trc", ".osim"))):  # Geometry(メッシュ約25MB/試行)は入れない
                cp(f, out / "imu3d" / f.relative_to(root / "imu3d"))
    for f in root.glob("summary_all.csv"):
        cp(f, out / f.name)
    for sd in sorted(p for p in root.iterdir() if p.is_dir() and p.name not in ("analysis", "imu3d")):
        if (sd / "summary.csv").exists():
            cp(sd / "summary.csv", out / sd.name / "summary.csv")
        for td in sorted(p for p in sd.iterdir() if p.is_dir()):
            names = TRIAL_FILES + (("kinematics.csv",) if with_kinematics else ())
            for fn in names:
                if (td / fn).exists():
                    cp(td / fn, out / "trials" / sd.name / td.name / fn)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "?"
    out.mkdir(parents=True, exist_ok=True)
    (out / "MANIFEST.txt").write_text(
        f"exported: {dt.datetime.now():%Y-%m-%d %H:%M}\ncode commit: {commit}\nfiles: {n}\nsize: {size / 1e6:.1f} MB\n"
        f"kinematics.csv included: {with_kinematics}\n", encoding="utf-8")
    log(f"{n} ファイル / {size / 1e6:.1f} MB → {out}\n次: git add results && git commit -m \"results\" && git push")
    return out


def run_import(src: Path | None = None, root: Path | None = None, log=print) -> int:
    src = src or ROOT / "results"
    root = root or ROOT / "output"
    if not src.is_dir():
        raise SystemExit(f"{src} がありません(共用PCで export して push し、git pull してください)")
    n = 0
    for sub, base in (("analysis", root / "analysis"), ("imu3d", root / "imu3d")):
        for f in (src / sub).rglob("*") if (src / sub).is_dir() else []:
            if f.is_file():
                _copy(f, base / f.relative_to(src / sub))
                n += 1
    for f in src.glob("*"):
        if f.is_file() and f.name == "summary_all.csv":
            _copy(f, root / f.name)
            n += 1
    for f in (src / "trials").rglob("*") if (src / "trials").is_dir() else []:
        if f.is_file():
            _copy(f, root / f.relative_to(src / "trials"))
            n += 1
    for sd in src.iterdir():
        if sd.is_dir() and (sd / "summary.csv").exists() and sd.name not in ("analysis", "imu3d", "trials"):
            _copy(sd / "summary.csv", root / sd.name / "summary.csv")
            n += 1
    mf = src / "MANIFEST.txt"
    log(f"{n} ファイルを {root} に戻しました" + (f"\n{mf.read_text(encoding='utf-8').strip()}" if mf.exists() else ""))
    return n
