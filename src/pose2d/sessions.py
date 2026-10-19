"""被験者/セッション固有の値(動画パス・fps・体格・キャリブ)を config/sessions.csv から読む。

パスはWindows形式(F:\\...)・WSL形式(/mnt/f/...)・相対パス(このフォルダ基準)のどれでも書け、
実行環境に合わせて自動変換する。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .calib import Calibration
from .config import ROOT

_WIN = re.compile(r"^([A-Za-z]):[\\/](.*)$")
_WSL = re.compile(r"^/mnt/([a-z])/(.*)$")


def resolve_path(p: str | Path) -> Path:
    s = str(p).strip()
    on_windows = os.name == "nt"
    if m := _WIN.match(s):
        s = s if on_windows else f"/mnt/{m[1].lower()}/{m[2].replace(chr(92), '/')}"
    elif m := _WSL.match(s):
        s = f"{m[1].upper()}:\\{m[2].replace('/', chr(92))}" if on_windows else s
    path = Path(s)
    return path if path.is_absolute() else ROOT / path


def _num(v, default=None):
    return default if pd.isna(v) or str(v).strip() == "" else float(v)


@dataclass
class Session:
    name: str
    subject: str
    height_m: float | None
    mass_kg: float | None
    kick_side: str
    fps: float
    video_dir: Path
    output_dir: Path
    calib_json: Path
    calib: Calibration | None  # sessions.csv のスケール直指定から作った場合のみ非None
    video_dirs: tuple[Path, ...] = ()  # video_dir を ; で区切って複数指定できる(成功/失敗フォルダなど)

    def load_calibration(self) -> Calibration:
        if self.calib is not None:
            return self.calib
        if not self.calib_json.exists():
            raise SystemExit(f"キャリブレーションがありません: {self.calib_json}\n"
                             f"  python run.py calib --session {self.name} --video <定規動画>")
        return Calibration.load(self.calib_json)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, comment="#")


def _dirs(v) -> tuple[Path, ...]:
    return tuple(resolve_path(x) for x in str(v).split(";") if x.strip())


def load_session(name: str) -> Session:
    path = ROOT / "config" / "sessions.csv"
    df = _read(path)
    local = ROOT / "config" / "sessions.local.csv"  # PC固有の上書き(git管理外)。空欄セルは上書きしない
    if local.exists():
        ov = _read(local)
        for _, o in ov.iterrows():
            hit = df["session"].str.strip() == o["session"].strip()
            if hit.any():
                for k, v in o.items():
                    if k != "session" and pd.notna(v) and str(v).strip():
                        if k not in df.columns:
                            df[k] = None
                        df.loc[hit, k] = v
            else:
                df = pd.concat([df, o.to_frame().T], ignore_index=True)
    row = df[df["session"].str.strip() == name]
    if row.empty:
        raise SystemExit(f"{path} に session '{name}' がありません(登録済み: {list(df['session'])})")
    r = row.iloc[0]
    g = lambda k: r.get(k) if k in r.index else None  # noqa: E731

    cal = None
    if all(_num(g(k)) is not None for k in ("origin_x", "origin_y", "scale_x_mm_px", "scale_y_mm_px")):
        cal = Calibration.from_scale((_num(g("origin_x")), _num(g("origin_y"))),
                                     _num(g("scale_x_mm_px")), _num(g("scale_y_mm_px")),
                                     bool(int(_num(g("mirror_x"), 0))))
    out = g("output_dir")
    return Session(
        name=name, subject=str(g("subject") or name), height_m=_num(g("height_m")),
        mass_kg=_num(g("mass_kg")), kick_side=str(g("kick_side") or "right").strip().lower(),
        fps=_num(g("fps")), video_dir=_dirs(g("video_dir"))[0], video_dirs=_dirs(g("video_dir")),
        output_dir=resolve_path(out) if pd.notna(out) and str(out).strip() else ROOT / "output" / name,
        calib_json=resolve_path(g("calib_json") if pd.notna(g("calib_json")) and str(g("calib_json")).strip()
                                else f"calib/{name}.json"),
        calib=cal)


def list_sessions() -> list[str]:
    """sessions.csv(+ sessions.local.csv)に登録された全セッション名。"""
    names = list(_read(ROOT / "config" / "sessions.csv")["session"].str.strip())
    local = ROOT / "config" / "sessions.local.csv"
    if local.exists():
        names += [n.strip() for n in _read(local)["session"] if n.strip() not in names]
    return names
