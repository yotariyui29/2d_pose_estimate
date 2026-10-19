"""使い方:
  python run.py calib --session S --video cali.avi [--points x,y x,y x,y]
  python run.py imu3d [--trials t1,t2] [--no-opensim]   (Xsens+2D→3D。docs/imu3d.md)
  python run.py factors / landing / sequential / mediation (着弾との照合・試行間の依存。README「着弾との突き合わせ」)
  python run.py run   --session S [--trial NAME] [--impact N] [--window T0 T1] [--redo] [--no-overlay] [--skip-done]
  (--session は all / a,b,c で複数被験者を一括処理)
"""
from __future__ import annotations

import argparse
import json
import sys

import cv2
import pandas as pd

from .calib import Calibration, pick_points_gui
from .config import ROOT, load_config
from .detect import detect_ball_only, detect_video
from .process import process
from .render import overlay_video, plot_summary
from .sessions import list_sessions, load_session, resolve_path

VIDEO_EXT = {".avi", ".mp4", ".mov", ".mkv"}


def cmd_calib(a) -> None:
    sess = load_session(a.session)
    cap = cv2.VideoCapture(str(resolve_path(a.video)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"動画を開けません: {a.video}")
    pts = [tuple(map(float, p.split(","))) for p in a.points] if a.points else pick_points_gui(frame)
    cal = Calibration(pts[0], pts[1], pts[2], a.x_len, a.y_len, a.mirror_x)
    cal.save(sess.calib_json)
    print(f"保存: {sess.calib_json}\n{cal.summary()}")
    if abs(cal.axis_angle_deg - 90) > 1.5 or abs(cal.mm_per_px_x - cal.mm_per_px_y) / cal.mm_per_px_x > 0.03:
        print("[警告] 軸の直交性/縦横スケール差が大きい。クリック位置を確認してください。")


def _impacts() -> dict[tuple[str, str], int]:
    p = ROOT / "config" / "impacts.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p, encoding="utf-8-sig")
    return {(str(r.session), str(r.trial)): int(r.impact_frame) for r in df.itertuples()}


def find_impact(tbl: dict[tuple[str, str], int], session: str, trial: str) -> int | None:
    """impacts.csv から試行のインパクトを探す。試行名が `kurogi_left_task2` でも `task2` でも同じ試行として扱う(動画名の違い対策)。"""
    if (session, trial) in tbl:
        return tbl[(session, trial)]
    hits = {v for (s, k), v in tbl.items() if s == session and (k.endswith("_" + trial) or trial.endswith("_" + k))}
    return hits.pop() if len(hits) == 1 else None


def _run_trial(v, sess, cal, fps, cfg, a, impact) -> dict:
    out = sess.output_dir / v.stem
    out.mkdir(parents=True, exist_ok=True)
    raw_p = out / "raw_px.csv"
    need = None  # 検出する範囲。インパクトと--windowが分かれば窓+余白だけ検出して時間を節約
    if a.window and impact is not None:
        m = int(0.1 * fps)  # フィルタ端効果対策の余白
        need = (max(0, impact + round(a.window[0] * fps) - m), impact + round(a.window[1] * fps) + m)
    raw = pd.read_csv(raw_p) if raw_p.exists() and not a.redo else None
    if raw is not None and need and not (raw["frame"].min() <= need[0] and raw["frame"].max() >= need[1]):
        raw = None  # キャッシュが必要範囲を覆っていない
    if raw is None:
        raw = detect_video(v, cfg.detect, frames=need, log=lambda m: print(m, flush=True))
        raw.to_csv(raw_p, index=False, float_format="%.3f")
    marker = out / "ball_checked"
    if (cfg.detect.ball_model and not marker.exists()
            and not ("ball_x" in raw and raw["ball_x"].notna().any())):
        # 姿勢だけ検出済み(ボール検出が無効だった)の場合、ボールだけ追加検出して保存
        lo, hi = int(raw["frame"].min()), int(raw["frame"].max())
        if impact is not None:  # インパクトが既知なら、ボール速度に必要な前後だけ検出して時間を節約
            lo, hi = max(lo, impact - round(0.05 * fps)), min(hi, impact + round(0.12 * fps))
        b = detect_ball_only(v, cfg.detect, (lo, hi), log=lambda m: print(m, flush=True))
        if b is not None:
            raw = raw.drop(columns=[c for c in ("ball_x", "ball_y", "ball_c") if c in raw]).merge(b, on="frame", how="left")
            raw.to_csv(raw_p, index=False, float_format="%.3f")
            marker.write_text("")
    kin, info = process(raw, cal, fps, cfg.process, cfg.detect.min_kp_conf, sess.kick_side, impact)
    win_frames = None
    if a.window:
        if "t_rel_impact_s" not in kin:
            raise RuntimeError("インパクト未検出のため --window を使えません(--impact か config/impacts.csv で指定)")
        t0, t1 = a.window
        kin = kin[kin["t_rel_impact_s"].between(t0 - 1e-9, t1 + 1e-9)].reset_index(drop=True)
        if kin.empty:
            raise RuntimeError("--window の区間にデータがありません")
        win_frames = (int(kin["frame"].iloc[0]), int(kin["frame"].iloc[-1]))
        info["window_s"] = [t0, t1]
    info.update(subject=sess.subject, height_m=sess.height_m, mass_kg=sess.mass_kg)
    kin.to_csv(out / "kinematics.csv", index=False, float_format="%.5f")
    (out / "summary.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_summary(kin, info, out / "plot.png", v.stem)
    if not a.no_overlay:
        overlay_video(v, raw, out / "overlay.mp4", info.get("impact_frame"), frames=win_frames)
    return info


def _run_session(name: str, a, cfg, imp_tbl) -> tuple[list[dict], list[tuple[str, str]]]:
    sess = load_session(name)
    cal = sess.load_calibration()
    fps = float(a.fps or sess.fps)  # 動画コンテナのfpsは再生用(例30)なので使わない
    print(f"\n##### session={name} subject={sess.subject} fps={fps} | {cal.summary()}", flush=True)
    dirs = [d for d in sess.video_dirs if d.is_dir()]
    for d in sess.video_dirs:
        if not d.is_dir():
            print(f"   [警告] video_dir が見つかりません(スキップ): {d}", flush=True)
    if not dirs:
        raise RuntimeError(f"video_dir が見つかりません: {sess.video_dir}")
    vids = sorted((v for d in dirs for v in d.iterdir() if v.suffix.lower() in VIDEO_EXT), key=lambda v: v.name)
    names = [v.stem for v in vids]
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise RuntimeError(f"動画名が複数のフォルダで重複しています(出力が上書きされる): {dup}")
    vids = [v for v in vids if not v.stem.lower().startswith("cali")]
    if a.trial:
        vids = [v for v in vids if v.stem == a.trial]
    if not vids:
        raise RuntimeError("対象動画がありません")
    rows, fails = [], []
    for i, v in enumerate(vids, 1):
        print(f"== [{name} {i}/{len(vids)}] {v.name}", flush=True)
        done = sess.output_dir / v.stem / "summary.json"
        if a.skip_done and done.exists():
            print("   処理済みのためスキップ")
            rows.append({"session": name, "trial": v.stem, **json.loads(done.read_text(encoding="utf-8"))})
            continue
        impact = a.impact if a.impact is not None else find_impact(imp_tbl, name, v.stem)
        if a.window and impact is None:  # 検出前に判断して無駄な処理を避ける
            print("   [スキップ] インパクト未登録(config/impacts.csv か --impact)")
            fails.append((f"{name}/{v.stem}", "インパクト未登録"))
            continue
        try:
            info = _run_trial(v, sess, cal, fps, cfg, a, impact)
        except Exception as e:  # 1試行の失敗で一括処理全体を止めない
            print(f"   [失敗] {type(e).__name__}: {e}", flush=True)
            fails.append((f"{name}/{v.stem}", f"{type(e).__name__}: {e}"))
            continue
        rows.append({"session": name, "trial": v.stem, **info})
        print("  ", {k: (round(x, 3) if isinstance(x, float) else x) for k, x in info.items()}, flush=True)
    if rows:
        sess.output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).drop(columns="session").to_csv(sess.output_dir / "summary.csv", index=False)
    return rows, fails


def cmd_run(a) -> None:
    cfg = load_config()
    names = list_sessions() if a.session == "all" else [s.strip() for s in a.session.split(",")]
    imp_tbl, rows, fails = _impacts(), [], []
    for n in names:
        try:
            r, f = _run_session(n, a, cfg, imp_tbl)
        except (RuntimeError, SystemExit) as e:
            print(f"[スキップ] session {n}: {e}", flush=True)
            fails.append((n, str(e)))
            continue
        rows += r
        fails += f
    if len(names) > 1 and rows:
        out = ROOT / "output" / "summary_all.csv"
        out.parent.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\n全体サマリ: {out}")
    print(f"\n完了{len(rows)} 試行 / 失敗・スキップ {len(fails)}")
    for k, why in fails:
        print(f"  - {k}: {why}")


def cmd_analyze(a) -> None:
    from .analyze import run_analysis

    run_analysis(sessions=[s.strip() for s in a.session.split(",")] if a.session else None,
                 out=resolve_path(a.out) if a.out else None)


def cmd_factors(a) -> None:
    from .kickfactors import run_factors

    run_factors(sessions=[s.strip() for s in a.session.split(",")] if a.session else None, xsens=not a.no_xsens)


def cmd_landing(a) -> None:
    from .landing import run_landing

    run_landing(sessions=[s.strip() for s in a.session.split(",")] if a.session else None)


def cmd_sequential(a) -> None:
    from .sequential import run_sequential

    ss = [s.strip() for s in a.session.split(",")] if a.session else None
    run_sequential(sessions=ss, n_perm=a.perm, seed=a.seed)
    run_sequential(sessions=ss, n_perm=a.perm, seed=a.seed, detrend_on=True)  # 試行とともに動く傾向を除いた版


def cmd_mediation(a) -> None:
    from .mediation import run_mediation

    run_mediation(sessions=[s.strip() for s in a.session.split(",")] if a.session else None, n_perm=a.perm, seed=a.seed)


def cmd_trialdep_figures(a) -> None:
    from .trialdep_figures import run_trialdep_figures

    run_trialdep_figures()


def cmd_score(a) -> None:
    from .accuracy_score import run_score

    run_score(sessions=[s.strip() for s in a.session.split(",")] if a.session else None, n_boot=a.boot, seed=a.seed)


def cmd_imu3d(a) -> None:
    from imu3d.run_all import main as run_all

    sys.argv = ["run.py imu3d", *a.rest]
    run_all()


def cmd_chain(a) -> None:
    from .chaincheck import run_chain

    run_chain()


def cmd_export(a) -> None:
    from .export import run_export

    run_export(with_kinematics=a.with_kinematics)


def cmd_import_results(a) -> None:
    from .export import run_import

    run_import()


def cmd_animate(a) -> None:
    from imu3d.animate import run_animate

    run_animate(trials=[s.strip() for s in a.trials.split(",")] if a.trials else None, speed=a.speed)


def cmd_contact(a) -> None:
    from .contact_sensitivity import run_contact

    run_contact()


def cmd_impacts(a) -> None:
    """インパクト未登録の試行について、自動検出したインパクトフレームを impacts.csv に追記できる形で表示する(overlay.mp4 で確認してから追記)。"""
    known = _impacts()
    rows = []
    for sj in sorted((ROOT / "output").glob("*/*/summary.json")):
        sess, trial = sj.parent.parent.name, sj.parent.name
        if find_impact(known, sess, trial) is not None or (a.session and sess not in a.session.split(",")):
            continue
        info = json.loads(sj.read_text(encoding="utf-8"))
        if info.get("impact_frame") is not None:
            rows.append((sess, trial, int(info["impact_frame"])))
    if not rows:
        print("インパクト未登録の処理済み試行はありません")
        return
    print("# 自動検出したインパクト(未登録の試行)。overlay.mp4 で確認し、正しければ config/impacts.csv に追記する")
    print("session,trial,impact_frame")
    for r in rows:
        print(",".join(map(str, r)))


PIPELINE_STEPS = ("run", "analyze", "factors", "landing", "sequential", "mediation", "trialdep", "contact", "imu3d", "animate")


def cmd_pipeline(a) -> None:
    """2D運動学 → 集計・図 → 要因表 → 着弾の図 → 接触モデルの感度解析 → 3D(Xsens) を順に実行する。1つの段階が失敗しても続ける。"""
    steps = [s.strip() for s in a.steps.split(",")]
    bad = [s for s in steps if s not in PIPELINE_STEPS]
    if bad:
        raise SystemExit(f"不明な段階: {bad}(使えるのは {PIPELINE_STEPS})")
    sess = a.session
    failed = []
    for s in PIPELINE_STEPS:
        if s not in steps:
            continue
        print(f"\n######## [{s}] ########", flush=True)
        try:
            if s == "run":
                cmd_run(argparse.Namespace(session=sess, trial=None, impact=None, fps=None, window=a.window, redo=a.redo,
                                           no_overlay=a.no_overlay, skip_done=a.skip_done))
            elif s == "analyze":
                cmd_analyze(argparse.Namespace(session=None if sess == "all" else sess, out=None))
            elif s == "factors":
                cmd_factors(argparse.Namespace(session=None if sess == "all" else sess, no_xsens=False))
            elif s == "landing":
                cmd_landing(argparse.Namespace(session=None))
            elif s == "sequential":
                cmd_sequential(argparse.Namespace(session=None, perm=10000, seed=0))
            elif s == "mediation":
                cmd_mediation(argparse.Namespace(session=None, perm=2000, seed=0))
            elif s == "trialdep":
                cmd_trialdep_figures(argparse.Namespace())
            elif s == "contact":
                cmd_contact(a)
            elif s == "imu3d":
                cmd_imu3d(argparse.Namespace(rest=["--no-opensim"] if a.no_opensim else []))
            elif s == "animate":
                cmd_animate(argparse.Namespace(trials=None, speed=0.2))
        except (Exception, SystemExit) as e:  # 1段階の失敗で全体を止めない
            print(f"[失敗] {s}: {type(e).__name__}: {e}", flush=True)
            failed.append(s)
    print(f"\n完了: {[s for s in steps if s not in failed]} / 失敗: {failed}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "imu3d":  # オプションをそのまま渡すため argparse の前に振り分ける
        cmd_imu3d(argparse.Namespace(rest=sys.argv[2:]))
        return
    ap = argparse.ArgumentParser(prog="pose2d")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calib", help="直角定規の3点クリックでキャリブレーション")
    c.add_argument("--session", required=True)
    c.add_argument("--video", required=True, help="定規を写したキャリブ動画(先頭フレーム使用)")
    c.add_argument("--points", nargs=3, metavar="X,Y", help="GUIを使わず 原点 横端 縦端 のpxを直接指定")
    c.add_argument("--x-len", type=float, default=1.1)
    c.add_argument("--y-len", type=float, default=1.0)
    c.add_argument("--mirror-x", action="store_true", help="x符号を反転(蹴り方向を+xにしたい場合)")
    c.set_defaults(fn=cmd_calib)
    r = sub.add_parser("run", help="検出→フィルタ→実寸運動学→出力")
    r.add_argument("--session", required=True, help="セッション名 / カンマ区切りで複数 / all=sessions.csvの全登録")
    r.add_argument("--trial")
    r.add_argument("--impact", type=int, help="インパクトフレーム手動指定(省略時はボール動き出しから自動)")
    r.add_argument("--fps", type=float)
    r.add_argument("--window", nargs=2, type=float, metavar=("T0", "T1"), help="出力区間(インパクト基準の秒) 例: --window -0.3 0.1")
    r.add_argument("--redo", action="store_true", help="検出をやり直す(通常はraw_px.csvを再利用)")
    r.add_argument("--no-overlay", action="store_true")
    r.add_argument("--skip-done", action="store_true", help="summary.json がある試行はスキップ(中断後の再開用)")
    r.set_defaults(fn=cmd_run)
    n = sub.add_parser("analyze", help="全試行の集計: ボール速度表・E_b/W_p/EI・有効質量・被験者別/被験者間の図")
    n.add_argument("--session", help="対象セッション(カンマ区切り。省略=output内の全員)")
    n.add_argument("--out", help="出力先(省略=output/analysis)")
    n.set_defaults(fn=cmd_analyze)
    f = sub.add_parser("factors", help="インパクト時の重心・姿勢・接触位相などの要因表(+config/landing.csvの着弾と相関)")
    f.add_argument("--session", help="対象セッション(カンマ区切り。省略=sessions.csvの全員)")
    f.add_argument("--no-xsens", action="store_true", help="Xsens(data/xsens)の要因を加えない")
    f.set_defaults(fn=cmd_factors)
    ld = sub.add_parser("landing", help="config/landing.csvの着弾位置をゴール平面の図にし、分布楕円の傾きを算出")
    ld.add_argument("--session", help="対象セッション(カンマ区切り。省略=landing.csvの全員)")
    ld.set_defaults(fn=cmd_landing)
    sq = sub.add_parser("sequential", help="試行間の依存(前の試行の着弾のずれが次にどう効くか): ラグ1自己相関と補正係数行列K(順序シャッフル検定つき)")
    sq.add_argument("--session", help="対象セッション(カンマ区切り。省略=landing.csvの全員)")
    sq.add_argument("--perm", type=int, default=10000, help="シャッフル回数(既定 10000)")
    sq.add_argument("--seed", type=int, default=0)
    sq.set_defaults(fn=cmd_sequential)
    md = sub.add_parser("mediation", help="前の試行のずれ→次の試行の運動学(補正のつまみ)→次の着弾の媒介解析(factors の後に実行)")
    md.add_argument("--session", help="対象セッション(カンマ区切り。省略=着弾のある全員)")
    md.add_argument("--perm", type=int, default=2000, help="シャッフル回数(既定 2000)")
    md.add_argument("--seed", type=int, default=0)
    md.set_defaults(fn=cmd_mediation)
    tf = sub.add_parser("trialdep-figures", help="試行間の依存(sequential・mediation)の結果を直感的に読める図にまとめる(sequential・mediation の後に実行)")
    tf.set_defaults(fn=cmd_trialdep_figures)
    sc = sub.add_parser("score", help="着弾からタスク精度スコア(CE・VE・TE)を定義して試算し、測定誤差の感度と新規計測の要件も出す")
    sc.add_argument("--session", help="対象セッション(カンマ区切り。省略=landing.csvの全員)")
    sc.add_argument("--boot", type=int, default=4000, help="ブートストラップ・乱数の回数")
    sc.add_argument("--seed", type=int, default=0)
    sc.set_defaults(fn=cmd_score)
    ip = sub.add_parser("impacts", help="インパクト未登録の試行の自動検出フレームを impacts.csv に追記できる形で表示")
    ip.add_argument("--session", help="対象セッション(カンマ区切り。省略=全員)")
    ip.set_defaults(fn=cmd_impacts)
    ch = sub.add_parser("chain", help="運動連鎖のタイミング(大腿-下腿のピーク差)と着弾zの関係の頑健性チェック(定義違い・速さの影響を除く偏相関)")
    ch.set_defaults(fn=cmd_chain)
    ex = sub.add_parser("export", help="解析結果(表・図・試行ごとの要約)を results/ に集める。git で別のPCへ渡す用")
    ex.add_argument("--with-kinematics", action="store_true", help="試行ごとの kinematics.csv も入れる(別のPCで factors/contact/chain を再実行できる。約70MB)")
    ex.set_defaults(fn=cmd_export)
    ir = sub.add_parser("import-results", help="results/ の内容を output/ に戻す(export の逆。別のPCで git pull したあと)")
    ir.set_defaults(fn=cmd_import_results)
    an = sub.add_parser("animate", help="3D運動学(imu3d の kinematics3d.csv)の骨格アニメーション mp4(英語版・日本語版)")
    an.add_argument("--trials", help="試行名(カンマ区切り。省略=3D運動学のある全試行)")
    an.add_argument("--speed", type=float, default=0.2, help="再生速度(実時間の倍率。既定0.2=5倍スロー)")
    an.set_defaults(fn=cmd_animate)
    ct = sub.add_parser("contact", help="足-ボール接触モデルの感度解析と、着弾のばらつきの分解(足先速度の向き vs 接触点のずれ)")
    ct.set_defaults(fn=cmd_contact)
    pl = sub.add_parser("pipeline", help="2D運動学→集計・図→要因表→着弾図→接触モデル→3D(Xsens)を順に実行")
    pl.add_argument("--session", default="all", help="all / カンマ区切り(2D運動学・集計の対象)")
    pl.add_argument("--steps", default=",".join(PIPELINE_STEPS), help=f"実行する段階(カンマ区切り): {PIPELINE_STEPS}")
    pl.add_argument("--window", nargs=2, type=float, metavar=("T0", "T1"), help="2D運動学の出力区間(秒)。3D(imu3d)には -0.9 0.5 が必要")
    pl.add_argument("--redo", action="store_true", help="2Dの検出からやり直す")
    pl.add_argument("--skip-done", action="store_true", help="処理済みの試行は2Dをやり直さない")
    pl.add_argument("--no-overlay", action="store_true")
    pl.add_argument("--no-opensim", action="store_true", help="3DでOpenSim出力を作らない")
    pl.set_defaults(fn=cmd_pipeline)
    im = sub.add_parser("imu3d", help="Xsens+2D→下半身3D運動学・OpenSim出力(旧3d_xsens_hscamera)。引数は imu3d/run_all.py と同じ")
    im.set_defaults(fn=cmd_imu3d)  # 実際の引数は main() 冒頭で振り分け: --trials kurogi_t3,kurogi_t4 [--source pose2d] [--no-opensim] [--no-align]
    a = ap.parse_args()
    a.fn(a)
    from . import i18n

    if i18n.untranslated():  # 日本語版の図で英語が残った文字(i18n.PHRASES に足す)
        print("[情報] 日本語版で未訳の図の文字:", i18n.untranslated(), flush=True)
