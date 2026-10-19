"""動画 → 生のキーポイント/ボール座標(px)。追跡・人物選択・ボール選択を含む。"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pandas as pd

from .config import DetectCfg
from .keypoints import COCO17, HALPE26, OUTPUT_KPS


class PoseBackend:
    names: list[str]

    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N人, K, 2) の座標と (N, K) の信頼度を返す。"""
        raise NotImplementedError


class RtmBackend(PoseBackend):
    names = HALPE26

    def __init__(self, cfg: DetectCfg):
        from rtmlib import BodyWithFeet

        import onnxruntime as ort

        has_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
        device = ("cuda" if has_cuda else "cpu") if cfg.device == "auto" else cfg.device
        if device == "cuda" and not has_cuda:
            print("[警告] onnxruntime にCUDAがありません(onnxruntime-gpuが必要)。CPUで実行します", flush=True)
            device = "cpu"
        self.m = BodyWithFeet(mode=cfg.rtm_mode, backend="onnxruntime",  # pose省略=modeに応じたHalpe26モデル
                              device=device, to_openpose=False)
        # DLL不足だとonnxruntimeは黙ってCPUに落ちるので、実際に使われるプロバイダを表示する
        used = self.m.pose_model.session.get_providers()[0]
        print(f"[情報] 姿勢推定デバイス: 要求={device} / 実際={used}", flush=True)
        if device == "cuda" and used != "CUDAExecutionProvider":
            print("[警告] GPUを使えていません(CPUで実行)。CUDA12/cuDNN9/MSVCランタイムを確認してください", flush=True)

    def __call__(self, frame):
        kps, sc = self.m(frame)
        return np.asarray(kps, float), np.asarray(sc, float)


class YoloBackend(PoseBackend):
    names = COCO17

    def __init__(self, cfg: DetectCfg):
        from ultralytics import YOLO

        self.m = YOLO(cfg.yolo_pose_model)

    def __call__(self, frame):
        r = self.m.predict(frame, verbose=False)[0]
        if r.keypoints is None or r.keypoints.xy.numel() == 0:
            return np.zeros((0, 17, 2)), np.zeros((0, 17))
        return r.keypoints.xy.cpu().numpy().astype(float), r.keypoints.conf.cpu().numpy().astype(float)


class OnnxBallDetector:
    """YOLOX(COCO学習済みONNX)でsports ballを検出。torch不要、onnxruntimeだけで動く。"""
    S = 640
    URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{}.onnx"

    def __init__(self, cfg: DetectCfg):
        import onnxruntime as ort

        path = Path(cfg.ball_model)
        if not path.suffix:
            path = Path.home() / ".cache" / "pose2d" / f"{cfg.ball_model}.onnx"
            if not path.exists():
                import urllib.request

                path.parent.mkdir(parents=True, exist_ok=True)
                print(f"[情報] ボール検出モデルをダウンロード: {self.URL.format(cfg.ball_model)}", flush=True)
                urllib.request.urlretrieve(self.URL.format(cfg.ball_model), path.with_suffix(".part"))
                path.with_suffix(".part").rename(path)
        prov = ort.get_available_providers()
        use = ["CUDAExecutionProvider"] if cfg.device != "cpu" and "CUDAExecutionProvider" in prov else []
        self.sess = ort.InferenceSession(str(path), providers=[*use, "CPUExecutionProvider"])
        self.cls, self.thr = cfg.ball_class_id, 0.15
        grids, strides = [], []
        for s in (8, 16, 32):
            n = self.S // s
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack([xv, yv], -1).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        self.grid, self.stride = np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)

    def __call__(self, frame) -> list[tuple[float, float, float]]:
        r = min(self.S / frame.shape[0], self.S / frame.shape[1])
        h, w = int(frame.shape[0] * r), int(frame.shape[1] * r)
        pad = np.full((self.S, self.S, 3), 114, np.uint8)
        pad[:h, :w] = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        o = self.sess.run(None, {"images": pad.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
        score = o[:, 4] * o[:, 5 + self.cls]
        k = np.flatnonzero(score > self.thr)
        if not len(k):
            return []
        xy, wh = (o[k, :2] + self.grid[k]) * self.stride[k], np.exp(o[k, 2:4]) * self.stride[k]
        boxes = np.column_stack([xy - wh / 2, wh]).tolist()
        keep = cv2.dnn.NMSBoxes(boxes, score[k].tolist(), self.thr, 0.45)
        return [(float(xy[i, 0] / r), float(xy[i, 1] / r), float(score[k][i])) for i in np.asarray(keep).ravel()]


class BallDetector:
    def __init__(self, cfg: DetectCfg):
        from ultralytics import YOLO

        self.m, self.cls = YOLO(cfg.ball_model), cfg.ball_class_id

    def __call__(self, frame) -> list[tuple[float, float, float]]:
        r = self.m.predict(frame, classes=[self.cls], verbose=False)[0]
        return [(float(x), float(y), float(c)) for (x, y, _, _), c in
                zip(r.boxes.xywh.cpu().numpy(), r.boxes.conf.cpu().numpy())]


_MODELS: dict = {}  # 一括処理でモデルを試行ごとに読み直さないためのキャッシュ


def make_backend(cfg: DetectCfg) -> PoseBackend:
    key = ("pose", cfg.backend, cfg.rtm_mode, cfg.device, cfg.yolo_pose_model)
    if key not in _MODELS:
        _MODELS[key] = {"rtmlib": RtmBackend, "yolo": YoloBackend}[cfg.backend](cfg)
    return _MODELS[key]


def make_ball_detector(cfg: DetectCfg, log=print) -> "BallDetector | None":
    key = ("ball", cfg.ball_backend, cfg.ball_model)
    if key not in _MODELS:
        try:
            _MODELS[key] = (OnnxBallDetector if cfg.ball_backend == "onnx" else BallDetector)(cfg)
        except Exception as e:  # torch/ultralytics が使えないPCでも姿勢推定だけは続ける
            log(f"[警告] ボール検出を無効化します(ball_x/yはNaN、ボール速度とインパクト自動検出なし): {e}")
            _MODELS[key] = None
    return _MODELS[key]


def iter_frames(video: Path) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                return
            yield f
    finally:
        cap.release()


def iter_frames_idx(video: Path, frames: tuple[int, int] | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """(frame番号, 画像)。frames=(開始,終了)なら範囲外は復号だけして読み飛ばす。"""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    try:
        f = 0
        if frames and frames[0] > 0:
            # MJPG等のフレーム内圧縮はシークが正確。それ以外は復号しながら読み飛ばす
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode("latin1").upper()
            if fourcc in ("MJPG", "MJPEG", "DIB ", "I420") and cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0]) \
                    and int(cap.get(cv2.CAP_PROP_POS_FRAMES)) == frames[0]:
                f = frames[0]
        while True:
            if frames and f > frames[1]:
                return
            if frames and f < frames[0]:
                if f % 500 == 0:
                    print(f"  (frame {f}/{frames[0]} まで読み飛ばし中)", flush=True)
                if not cap.grab():
                    return
            else:
                ok, img = cap.read()
                if not ok:
                    return
                yield f, img
            f += 1
    finally:
        cap.release()


def _pick_person(kps, sc, prev_center, roi, min_conf):
    """前フレームの人物中心に最も近い人物(初回は最大の人物)を選ぶ。"""
    best, best_cost = None, np.inf
    for i in range(len(kps)):
        ok = sc[i] >= min_conf
        if ok.sum() < 5:
            continue
        pts = kps[i][ok]
        c = pts.mean(0)
        if roi is not None and not (roi[0] <= c[0] <= roi[2] and roi[1] <= c[1] <= roi[3]):
            continue
        size = np.ptp(pts, axis=0).max()
        cost = np.linalg.norm(c - prev_center) if prev_center is not None else -size
        if cost < best_cost:
            best, best_cost = i, cost
    return best


class BallTracker:
    """候補のうち、前フレーム位置から gate_px 以内で最も信頼度が高いものを採る。"""

    def __init__(self, gate_px: float):
        self.gate, self.prev, self.miss = gate_px, None, 0

    def update(self, cands) -> tuple[float, float, float]:
        if self.prev is not None:
            cands = [c for c in cands if np.hypot(c[0] - self.prev[0], c[1] - self.prev[1]) <= self.gate]
            self.miss = 0 if cands else self.miss + 1
            if self.miss > 30:  # 見失い続けたら追跡をリセット
                self.prev, self.miss = None, 0
        if not cands:
            return np.nan, np.nan, np.nan
        bx, by, bc = max(cands, key=lambda c: c[2])
        self.prev = (bx, by)
        return bx, by, bc


def detect_ball_only(video: Path, cfg: DetectCfg, frames: tuple[int, int], log=print) -> pd.DataFrame | None:
    """姿勢推定済みの範囲に、ボールだけを追加検出する。検出器が使えなければNone。"""
    ball = make_ball_detector(cfg, log) if cfg.ball_model else None
    if ball is None:
        return None
    tracker, rows, t0 = BallTracker(cfg.ball_gate_px), [], time.time()
    for f, frame in iter_frames_idx(video, frames):
        rows.append({"frame": f, **dict(zip(("ball_x", "ball_y", "ball_c"), tracker.update(ball(frame))))})
        if len(rows) % 100 == 0:
            log(f"  ball frame {f}  ({len(rows)} done, {time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)


def detect_video(video: Path, cfg: DetectCfg, roi: tuple[float, float, float, float] | None = None,
                 max_frames: int | None = None, log=print,
                 frames: tuple[int, int] | None = None) -> pd.DataFrame:
    """1動画を処理して生データ(px)のDataFrameを返す。列: frame, <kp>_x/_y/_c, ball_x/_y/_c"""
    pose = make_backend(cfg)
    ball = make_ball_detector(cfg, log) if cfg.ball_model else None
    idx = {n: i for i, n in enumerate(pose.names)}
    missing = [k for k in OUTPUT_KPS if k not in idx]
    if missing:
        log(f"[注意] backend={cfg.backend} は {missing} を出力しない(NaNになる)")

    rows, prev_c, tracker = [], None, BallTracker(cfg.ball_gate_px)
    t_start = time.time()
    for f, frame in iter_frames_idx(video, frames):
        if max_frames and f >= max_frames:
            break
        row: dict[str, float] = {"frame": f}
        kps, sc = pose(frame)
        p = _pick_person(kps, sc, prev_c, roi, cfg.min_kp_conf) if len(kps) else None
        for k in OUTPUT_KPS:
            if p is not None and k in idx:
                row[f"{k}_x"], row[f"{k}_y"] = kps[p, idx[k]]
                row[f"{k}_c"] = sc[p, idx[k]]
            else:
                row[f"{k}_x"] = row[f"{k}_y"] = row[f"{k}_c"] = np.nan
        if p is not None:
            ok = sc[p] >= cfg.min_kp_conf
            prev_c = kps[p][ok].mean(0)
        bx, by, bc = tracker.update(ball(frame)) if ball is not None else (np.nan,) * 3
        row.update(ball_x=bx, ball_y=by, ball_c=bc)
        rows.append(row)
        if len(rows) % 20 == 0:
            log(f"  frame {f}  ({len(rows)} done, {time.time() - t_start:.0f}s)")
    return pd.DataFrame(rows)
