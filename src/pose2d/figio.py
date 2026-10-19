"""図の保存。Windows のウイルス対策やドライブの一時的な不調で OSError [Errno 22] が出ることがあるため、
再試行し、それでも失敗したら一時フォルダへ保存して処理を止めない。

英語版 `x.png` を保存したあと、図の文字を日本語にした `x_ja.png` も保存する(pose2d.i18n。環境変数 POSE2D_NO_JA=1 で無効)。
日本語化は図をその場で書き換えるので、save_figure のあとで図を再利用しないこと(呼び出し側は保存後に close している)。
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

from . import i18n


def _save(fig, path: Path, retries: int, **kwargs) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    err: OSError | None = None
    for _ in range(retries):
        try:
            fig.savefig(path, **kwargs)
            return path
        except OSError as e:
            err = e
            time.sleep(1.0)
    alt = Path(tempfile.gettempdir()) / "pose2d_figs" / path.name
    alt.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(alt, **kwargs)
    print(f"[警告] 図を保存できません({err})。代わりに {alt} へ保存しました: {path}", flush=True)
    return alt


def ja_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_ja{path.suffix}")


def save_figure(fig, path: Path, retries: int = 3, ja: bool = True, **kwargs) -> Path:
    path = Path(path)
    out = _save(fig, path, retries, **kwargs)
    if ja and i18n.enabled():
        if i18n.to_japanese(fig):
            kw = dict(kwargs)
            kw.setdefault("bbox_inches", "tight")  # 日本語化で文字の長さが変わるので、はみ出しを避ける
            import matplotlib

            with matplotlib.rc_context({"font.family": [i18n.japanese_font()]}):  # 保存時に作り直される文字(目盛りなど)も日本語フォントにする
                _save(fig, ja_path(path), retries, **kw)
        elif not getattr(save_figure, "_warned", False):
            save_figure._warned = True  # type: ignore[attr-defined]
            print("[警告] 日本語フォントが見つからないため、日本語版の図(_ja.png)は作りません(IPAexGothic / Meiryo / Yu Gothic など)", flush=True)
    return out
