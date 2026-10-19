"""図の日本語化の検証: python tests/test_i18n.py (WSL venv)"""
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pose2d import i18n
from pose2d.figio import ja_path, save_figure


def test_translate() -> None:
    assert i18n.translate("Time from impact (s)") == "インパクトからの時間 (s)"
    assert i18n.translate("thigh") == "大腿"
    assert i18n.translate("mean (n=7)") == "平均 (n=7)"
    assert i18n.translate("foot velocity elevation (+/-2.5 deg)") == "足先速度の仰角 (±2.5°)"
    assert "傾き +40°" in i18n.translate("95% ellipse (tilt in this plot +40°)")
    assert i18n.translate("kurogi") == "kurogi"  # 被験者名は変えない


def test_save_both() -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="thigh")
    ax.set_xlabel("Time from impact (s)")
    ax.legend()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.png"
        save_figure(fig, p, dpi=60)
        plt.close(fig)
        assert p.exists()
        if i18n.japanese_font():
            assert ja_path(p).exists()
        else:
            print("日本語フォントなし: _ja.png は省略")


if __name__ == "__main__":
    test_translate()
    test_save_both()
    print("OK")
