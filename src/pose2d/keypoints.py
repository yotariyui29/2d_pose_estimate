"""統一キーポイント名。バックエンドごとのインデックスを共通名に対応付ける。"""
from __future__ import annotations

COCO17 = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
          "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
          "left_wrist", "right_wrist", "left_hip", "right_hip",
          "left_knee", "right_knee", "left_ankle", "right_ankle"]

# Halpe26 = COCO17 + head, neck, hip(pelvis), 足先/踵
HALPE26 = COCO17 + ["head", "neck", "pelvis", "left_big_toe", "right_big_toe",
                    "left_small_toe", "right_small_toe", "left_heel", "right_heel"]

# CSV に出力するキーポイント(目・耳は除く。鼻と頭頂は重畳表示用に残す)
OUTPUT_KPS = [k for k in HALPE26 if k not in ("left_eye", "right_eye", "left_ear", "right_ear")]

# 体節: (名前, 近位, 遠位)
SEGMENTS = [
    ("thigh_R", "right_hip", "right_knee"), ("thigh_L", "left_hip", "left_knee"),
    ("shank_R", "right_knee", "right_ankle"), ("shank_L", "left_knee", "left_ankle"),
    ("foot_R", "right_heel", "right_big_toe"), ("foot_L", "left_heel", "left_big_toe"),
    ("trunk", "pelvis", "neck"),
]
