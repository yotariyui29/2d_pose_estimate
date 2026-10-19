"""文献に基づく追加チェック(既存データの再解析)。

既存の結果(output/analysis, results/ など)は読むだけで、書き出しは output/lit_followup/ だけ(現在の解析結果と混ざらない)。
実行: リポジトリ直下で `PYTHONPATH=src python -m lit_followup.<モジュール名>`
"""
import os

os.environ.setdefault("MPLBACKEND", "Agg")
