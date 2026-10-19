"""sessions.csv 読み込み・パス変換・旧スケール方式の検証: python tests/test_sessions.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pose2d.calib import Calibration
from pose2d.sessions import load_session, resolve_path

s = load_session("kataoka")
cal = s.load_calibration()
assert (s.mass_kg, s.kick_side, s.fps) == (70, "right", 1000)
assert abs(cal.mm_per_px_x - 3.5) < 1e-9 and abs(cal.mm_per_px_y - 3.5) < 1e-9
# 旧式: x_m=(px-ox)*sx/1000, y_m=(oy-py)*sy/1000
p = np.array([[1379.0, 644.0]])
assert np.allclose(cal.px_to_m(p), [[0.35, 0.35]]), cal.px_to_m(p)
print(resolve_path("/mnt/f/yamada/x.avi"), resolve_path(r"F:\yamada\x.avi"), resolve_path("data/a"))
print("OK")

# sessions.local.csv による上書き(video_dirのみ)。他の列は共通設定のまま
local = Path(__file__).resolve().parents[1] / "config" / "sessions.local.csv"
if local.exists():
    k = load_session("kurogi")
    print("local override:", k.video_dir, k.fps, k.kick_side)
    assert k.fps == 2000 and k.kick_side == "left"
