"""直角定規(既定 1.1 m × 1.0 m)による ピクセル→実寸(m) 変換。

原点O・横端X(1.1 m)・縦端Y(1.0 m)の3点から、2×2アフィン行列を求める。
軸のわずかな傾き・非直交・ピクセル縦横比の違いも吸収される。
実寸座標は x: 横(定規X方向)、y: 上向き(定規Y方向)。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class Calibration:
    origin_px: tuple[float, float]
    x_px: tuple[float, float]
    y_px: tuple[float, float]
    x_len_m: float = 1.1
    y_len_m: float = 1.0
    mirror_x: bool = False  # True: x符号反転(蹴り方向を+xに揃える用)

    def __post_init__(self) -> None:
        o = np.asarray(self.origin_px, float)
        ex = (np.asarray(self.x_px, float) - o) / self.x_len_m  # px/m
        ey = (np.asarray(self.y_px, float) - o) / self.y_len_m
        self._o = o
        self._inv = np.linalg.inv(np.column_stack([ex, ey]))
        self.mm_per_px_x = 1000.0 / np.linalg.norm(ex)
        self.mm_per_px_y = 1000.0 / np.linalg.norm(ey)
        cosang = float(ex @ ey / (np.linalg.norm(ex) * np.linalg.norm(ey)))
        self.axis_angle_deg = float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))

    @classmethod
    def from_scale(cls, origin_px, mm_per_px_x: float, mm_per_px_y: float, mirror_x: bool = False) -> "Calibration":
        """旧方式(原点px + mm/px、軸は画像に平行、y上向き)から作る。"""
        ox, oy = origin_px
        return cls((ox, oy), (ox + 1000.0 / mm_per_px_x, oy), (ox, oy - 1000.0 / mm_per_px_y),
                   1.0, 1.0, mirror_x)

    def px_to_m(self, xy_px: np.ndarray) -> np.ndarray:
        """(..., 2) のpx座標を(..., 2)のm座標へ。NaNは保持。"""
        p = (np.asarray(xy_px, float) - self._o) @ self._inv.T
        if self.mirror_x:
            p[..., 0] *= -1
        return p

    def px_vec_to_m(self, v_px: np.ndarray) -> np.ndarray:
        """原点オフセット無しのベクトル変換。"""
        p = np.asarray(v_px, float) @ self._inv.T
        if self.mirror_x:
            p[..., 0] *= -1
        return p

    def summary(self) -> str:
        return (f"{self.mm_per_px_x:.3f} mm/px (x), {self.mm_per_px_y:.3f} mm/px (y), "
                f"軸のなす角 {self.axis_angle_deg:.2f}° (90°に近いほど良い)")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def pick_points_gui(image_bgr: np.ndarray) -> list[tuple[float, float]]:
    """画像上で 原点→横端→縦端 の順に3点クリック。Enterで確定、Backspaceで1点戻す。"""
    import cv2

    pts: list[tuple[float, float]] = []
    labels = ["1: origin", "2: x end (1.1m)", "3: y end (1.0m)"]
    win = "calibration: click origin -> x end -> y end (Enter=OK, Backspace=undo, Esc=quit)"

    def redraw():
        img = image_bgr.copy()
        for i, (x, y) in enumerate(pts):
            cv2.drawMarker(img, (int(x), int(y)), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
            cv2.putText(img, labels[i], (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(win, img)

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 3:
            pts.append((float(x), float(y)))
            redraw()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    redraw()
    while True:
        k = cv2.waitKey(50) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            raise SystemExit("キャリブレーション中断")
        if k == 8 and pts:
            pts.pop(); redraw()
        if k in (13, 10) and len(pts) == 3:
            break
    cv2.destroyAllWindows()
    return pts
