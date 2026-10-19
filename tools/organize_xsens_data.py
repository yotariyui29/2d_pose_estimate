"""data/xsens の整理(1回きりの移動 + 目録作成)。ファイルは削除せず、移動のみ。

旧: data/xsens/<日付>_<被験者>/[left|right]_shoot/{Xsens|xsens|EMG|powerlab|acc}/{tN|taskN}/{stop,task}
新: data/xsens/<被験者>/<日付>/[left|right]_shoot/{xsens|emg|mvc|powerlab|acc|notes}/taskN/{stop,task}
    (tN → taskN、種別フォルダは小文字、コピー/テスト等は _unsorted/ へ退避、側のない日付直下のMVC/powerlabは日付直下のまま)

  python tools/organize_xsens_data.py            # 計画の表示のみ(dry-run)
  python tools/organize_xsens_data.py --apply    # 実行。data/xsens/_moves.csv(旧→新)と INVENTORY.csv を出力
元に戻すときは _moves.csv の新→旧を逆に移動する。
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "xsens"
SIDES = {"left_shoot", "right_shoot"}
KINDS = {"xsens", "emg", "mvc", "powerlab", "acc"}
SESSION = re.compile(r"^(\d{8})_([A-Za-z]+)$")
TRIAL = re.compile(r"^(?:t|task)(\d+)$", re.I)
JUNK_FILE_EXT = {".py", ".sh", ".mp4"}


def map_parts(parts: tuple[str, ...]) -> tuple[str, ...]:
    """旧パス(ROOT基準のファイル)→新パス。対象外(整理済み・_で始まる)は変更しない。"""
    m = SESSION.match(parts[0])
    if not m or len(parts) < 2:
        return parts
    date, subj = m[1], m[2].lower()
    rest = list(parts[1:])
    out = [subj, date]
    if rest[0] in SIDES:
        out.append(rest.pop(0))
    if not rest:
        return tuple(out)
    if len(rest) == 1:  # 側/日付直下のファイル(実験メモなど)
        return tuple(out + ["notes", rest[0]])
    if rest[0].lower() in KINDS:
        kind = rest.pop(0).lower()
        out.append(kind)
        if kind == "xsens" and rest:
            first = rest[0]
            tm = TRIAL.match(first)
            if len(rest) == 1:  # xsensフォルダ直下のファイル(タイミング表など)
                return tuple(out + rest)
            if tm:
                out.append(f"task{int(tm[1])}")
                rest.pop(0)
                if rest[0] == "animations" or Path(rest[0]).suffix.lower() in JUNK_FILE_EXT:
                    return tuple(out[:-1] + ["_unsorted", f"task{int(tm[1])}_extra"] + rest)
            else:  # 「新しいフォルダー」「1 - コピー」「テスト」など試行名でないもの
                return tuple(out[:-1] + ["_unsorted", "xsens_" + first] + rest[1:])
        return tuple(out + rest)
    return tuple(out + ["_other"] + rest)  # 想定外の構造は _other に集める(後で確認)


def plan() -> list[tuple[Path, Path]]:
    moves = []
    for f in sorted(p for p in ROOT.rglob("*") if p.is_file()):
        rel = f.relative_to(ROOT).parts
        if rel[0].startswith("_") or rel[-1] in ("_moves.csv", "INVENTORY.csv"):
            continue
        new = map_parts(rel)
        if new != rel:
            moves.append((f, ROOT.joinpath(*new)))
    dst = Counter(d for _, d in moves)
    dup = [d for d, n in dst.items() if n > 1]
    if dup:
        raise SystemExit(f"移動先が重複: {dup[:5]}")
    exist = [d for _, d in moves if d.exists()]
    if exist:
        raise SystemExit(f"移動先が既に存在: {exist[:5]}")
    return moves


def inventory() -> list[dict]:
    rows: dict[tuple, list] = defaultdict(lambda: [0, 0])
    for f in ROOT.rglob("*"):
        if not f.is_file() or f.name in ("_moves.csv", "INVENTORY.csv"):
            continue
        p = f.relative_to(ROOT).parts
        if len(p) < 3:
            continue
        side = p[2] if p[2] in SIDES else ""
        k = p[3] if side and len(p) > 3 else (p[2] if not side and len(p) > 2 else "")
        trial = next((x for x in p if TRIAL.match(x)), "")
        key = (p[0], p[1], side, k, trial)
        rows[key][0] += 1
        rows[key][1] += f.stat().st_size
    return [dict(subject=a, date=b, side=c, kind=d, trial=e, n_files=n, size_mb=round(s / 1e6, 1))
            for (a, b, c, d, e), (n, s) in sorted(rows.items())]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    moves = plan()
    by = Counter((s.relative_to(ROOT).parts[0], d.relative_to(ROOT).parts[:2]) for s, d in moves)
    print(f"移動するファイル {len(moves)} 件")
    for (old, new), n in sorted(by.items()):
        print(f"  {old} → {'/'.join(new)}  ({n})")
    uns = sorted({"/".join(d.relative_to(ROOT).parts[:5]) for _, d in moves if "_unsorted" in d.relative_to(ROOT).parts or "_other" in d.relative_to(ROOT).parts})
    print(f"_unsorted/_other に退避: {len(uns)} 箇所")
    for u in uns[:40]:
        print("   ", u)
    if not a.apply:
        print("\n(dry-run) 実行は --apply")
        return
    with open(ROOT / "_moves.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["old", "new"])
        for s, d in moves:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(s), str(d))
            w.writerow([s.relative_to(ROOT).as_posix(), d.relative_to(ROOT).as_posix()])
    for d in sorted((p for p in ROOT.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)):  # 空になった旧フォルダを削除
        if not any(d.iterdir()):
            d.rmdir()
    inv = inventory()
    with open(ROOT / "INVENTORY.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(inv[0]))
        w.writeheader()
        w.writerows(inv)
    print(f"完了: {ROOT / '_moves.csv'}, {ROOT / 'INVENTORY.csv'}")


if __name__ == "__main__":
    main()
