from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DetectCfg:
    backend: str = "rtmlib"
    rtm_mode: str = "performance"
    device: str = "auto"  # auto: CUDAが使えればGPU / cpu / cuda
    yolo_pose_model: str = "yolov8x-pose.pt"
    ball_backend: str = "onnx"  # onnx: onnxruntimeのみ(torch不要) / ultralytics: torch必須
    ball_model: str = "yolox_m"  # onnx: yolox_s|m|l|x(初回に~/.cache/pose2dへ自動DL) / ultralytics: yolov8x.pt等
    ball_class_id: int = 32
    min_kp_conf: float = 0.3
    ball_gate_px: float = 150.0


@dataclass
class ProcessCfg:
    cutoff_hz: float | str = 15.0
    filter_order: int = 4
    max_gap_frames: int = 30
    spike_mad: float = 6.0
    pad_frames: int = 50


@dataclass
class Config:
    detect: DetectCfg = field(default_factory=DetectCfg)
    process: ProcessCfg = field(default_factory=ProcessCfg)


def _fill(cls, d: dict):
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"{cls.__name__}: 未知の設定キー {sorted(unknown)}")
    return cls(**d)


def load_config(path: Path | None = None) -> Config:
    path = path or ROOT / "config" / "pipeline.toml"
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return Config(_fill(DetectCfg, raw.get("detect", {})), _fill(ProcessCfg, raw.get("process", {})))
