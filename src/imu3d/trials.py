"""設定(config/)・PCごとのパス・2D/Xsens入力の読み込み。

- config/trials.csv   … 1行=1試行（Xsensフォルダ、インパクト-0.9sのXsens時刻 start_s、2Dの場所など）。コミットする
- config/sensors.csv  … センサーIDと部位の対応（sensor_set ごと）
- config/paths.toml   … このPCのデータの場所。別PCでは config/paths.local.toml（git管理外）に差分だけ書く
パスは Windows形式(C:\\...)・WSL形式(/mnt/c/...)・相対(リポジトリ基準)のどれでも書ける。
"""
from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config"
WINDOW = (-0.9, 0.5)  # インパクト基準の解析区間[s]（Xsensのstart/end）

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


def load_paths(config_dir: Path = CONFIG) -> dict[str, Path]:
    d: dict = {}
    for name in ("paths.toml", "paths.local.toml"):
        f = config_dir / name
        if f.exists():
            d.update(tomllib.loads(f.read_text(encoding="utf-8")))
    return {k: resolve_path(v) for k, v in d.items()}


class Trial2D:
    """高速度カメラの2D（x右・y上[m]）。t はインパクト基準の時刻[s]。点名: MP(つま先), HEEL, ANKLE, KNEE, HIP（キック脚）。"""

    def __init__(self, t: np.ndarray, pts: dict[str, np.ndarray], source: str):
        self.t, self.pts, self.source = np.asarray(t, float), pts, source

    @classmethod
    def from_kinovea(cls, path: Path, fps: float, flip_x: bool = False) -> "Trial2D":
        """Kinovea書き出し(ヘッダなし): frame(インパクト=0), MP, HEEL, ANKLE, KNEE, HIP の x,y [mm]。fpsは書き出しの時間刻み。
        flip_x: Kinovea座標のx軸が実際の画像と逆向きのとき反転（実際の画像の右向きを+xにそろえる）。"""
        a = pd.read_csv(path, header=None, encoding="utf-8-sig").to_numpy(float)
        names = ["MP", "HEEL", "ANKLE", "KNEE", "HIP"]
        pts = {n: a[:, 1 + 2 * i:3 + 2 * i] / 1000.0 * (np.array([-1.0, 1.0]) if flip_x else 1.0) for i, n in enumerate(names)}
        return cls(a[:, 0] / fps, pts, "kinovea")

    @classmethod
    def from_pose2d(cls, path: Path, kick: str) -> "Trial2D":
        """2d_pose_estimate の kinematics.csv（m, t_rel_impact_s）。MP=big_toe。"""
        df = pd.read_csv(path)
        side = "left" if kick == "L" else "right"
        cols = [f"{side}_{j}_{a}" for j in ("big_toe", "heel", "ankle", "knee", "hip") for a in ("x", "y")]
        df[cols] = df[cols].interpolate(limit=100, limit_area="inside")  # 検出の欠け(2Dで補間しきれなかった0.03〜0.1秒程度)を直線補間
        xy = lambda j: df[[f"{side}_{j}_x", f"{side}_{j}_y"]].to_numpy(float)
        pts = {"MP": xy("big_toe"), "HEEL": xy("heel"), "ANKLE": xy("ankle"), "KNEE": xy("knee"), "HIP": xy("hip")}
        return cls(df["t_rel_impact_s"].to_numpy(float), pts, "pose2d")

    def at(self, tau: np.ndarray) -> dict[str, np.ndarray]:
        return {n: np.stack([np.interp(tau, self.t, p[:, i]) for i in (0, 1)], 1) for n, p in self.pts.items()}

    def lengths(self) -> dict[str, float]:
        L = lambda a, b: float(np.percentile(np.linalg.norm(self.pts[a] - self.pts[b], axis=1), 95))
        return {"thigh": L("HIP", "KNEE"), "shank": L("KNEE", "ANKLE"), "foot": L("ANKLE", "MP")}

    def foot_pitch(self) -> float:
        """足部ベクトルの水平からの下向き角δ（足裏接地フレームの中央値。無ければ0）。"""
        p = self.pts
        flat = np.abs(p["MP"][:, 1] - p["HEEL"][:, 1]) < 0.015
        if flat.sum() < 3:
            return 0.0
        v = p["MP"][flat] - p["ANKLE"][flat]
        return float(np.median(np.arctan2(-v[:, 1], np.abs(v[:, 0]))))


@dataclass
class Trial:
    name: str
    subject: str
    kick: str  # "L" / "R"
    xsens_dir: Path  # <xsens_root>/<xsens_subdir>（中に stop/ と task/）
    start_s: float  # インパクト-0.9s のXsens時刻（task先頭=0）
    sensors: dict[str, str]
    pose2d: Path | None
    kinovea: Path | None
    kinovea_fps: float
    kinovea_flip_x: bool
    note: str = ""

    @property
    def impact_s(self) -> float:
        return self.start_s + 0.9

    def load_2d(self, source: str = "auto") -> Trial2D | None:
        """source=auto: 使えるソースのうち、解析区間(WINDOW)との重なりが最長のもの（同点なら pose2d 優先）。"""
        loaders = {"pose2d": lambda: Trial2D.from_pose2d(self.pose2d, self.kick) if self.pose2d is not None and self.pose2d.exists() else None,
                   "kinovea": lambda: Trial2D.from_kinovea(self.kinovea, self.kinovea_fps, self.kinovea_flip_x)
                   if self.kinovea is not None and self.kinovea.exists() else None}
        cands = [t for s in (["pose2d", "kinovea"] if source == "auto" else [source]) if (t := loaders[s]()) is not None]
        overlap = lambda t: min(WINDOW[1], t.t.max()) - max(WINDOW[0], t.t.min())
        return max(cands, key=overlap) if cands else None


def load_trials(paths: dict[str, Path] | None = None, config_dir: Path = CONFIG) -> dict[str, Trial]:
    paths = paths or load_paths(config_dir)
    sens = pd.read_csv(config_dir / "sensors.csv", dtype=str)
    sets = {s: dict(zip(g["part"], g["id"])) for s, g in sens.groupby("sensor_set")}
    out: dict[str, Trial] = {}
    for r in pd.read_csv(config_dir / "trials.csv", dtype=str, keep_default_na=False).itertuples():
        p2 = paths["pose2d_root"] / r.subject / r.pose2d_trial / "kinematics.csv" if r.pose2d_trial and "pose2d_root" in paths else None
        kv = paths["kinovea_root"] / r.kinovea_csv if r.kinovea_csv and "kinovea_root" in paths else None
        xp = getattr(r, "xsens_path", "")  # 被験者ごとにXsensの場所が違うので、あれば xsens_root/xsens_subdir より優先する
        xdir = resolve_path(xp) if xp else paths["xsens_root"] / r.xsens_subdir
        out[r.trial] = Trial(r.trial, r.subject, r.kick, xdir, float(r.start_s), sets[r.sensor_set],
                             p2, kv, float(r.kinovea_fps or 1000), r.kinovea_flip_x == "1", r.note)
    return out
