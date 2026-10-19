"""共通: 出力先、図の日本語辞書(メモリ上だけで追加。pose2d/i18n.py は編集しない)、表の場所。"""
from __future__ import annotations

from pathlib import Path

from pose2d import i18n
from pose2d.config import ROOT

OUT_DIR = ROOT / "output" / "lit_followup"

i18n.PHRASES.update({
    "Landing y, kick side + (m)": "着弾 y, 蹴り足側が+ (m)",
    "Landing z (m)": "着弾 z (m)",
    "observed 95% ellipse": "実測の95%楕円",
    "same SD, no covariation": "同じSD・共変動なし",
    "bootstrap tilt of the long axis (deg)": "長軸の傾きのブートストラップ分布 (deg)",
    "count": "回数",
    "Low-pass cutoff (Hz)": "ローパスのカットオフ (Hz)",
    "Toe-velocity elevation at impact (deg)": "インパクト直前のつま先速度の仰角 (deg)",
    "Ball elevation - toe-velocity elevation (deg)": "ボール出射仰角 − つま先速度の仰角 (deg)",
    "Toe-speed peak time vs impact (ms)": "つま先速度ピークの時刻(インパクト基準, ms)",
    "Shank-peak minus thigh-peak time (ms)": "下腿ピーク − 大腿ピークの時刻差 (ms)",
    "Time from impact (ms)": "インパクトからの時間 (ms)",
    "Toe speed (m/s)": "つま先の速さ (m/s)",
    "axis ratio": "軸比",
    "cutoff": "カットオフ",
})
i18n._ORDERED[:] = sorted(i18n.PHRASES.items(), key=lambda kv: -len(kv[0]))


def tables_dir() -> Path:
    """kick_factors.csv などの集計表。この PC の output/ を優先し、無ければ import-results 前の results/ を使う。"""
    for base in (ROOT / "output", ROOT / "results"):
        if (base / "analysis" / "tables" / "kick_factors.csv").exists():
            return base / "analysis" / "tables"
    raise SystemExit("kick_factors.csv がありません(python run.py factors か import-results を先に実行)")
