"""インパクト時の姿勢・接触位相など「ボールの飛び方に効きそうな要因」を試行ごとに算出し、着弾(y,z)と突き合わせる。
  python run.py factors [--session S]

入力: output/<session>/<trial>/{kinematics.csv, summary.json}、config/landing.csv(任意)
出力: output/analysis/tables/kick_factors.csv (+ 着弾があれば kick_factors_corr.csv)
座標: 蹴り方向が +x、上が +y(m)。角度は度。着弾(y,z)はゴール平面(y>0 は被験者から見て右側、z>0 は上)。
landing.csv の y は、蹴り足側が + になるよう y_kickside_m に変換して出力する(右足なら y そのまま、左足なら符号反転)。
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT
from .figio import save_figure
from .mechanics2d import BALL_WINDOW_S, _p, analyze_trial, kick_sign
from .sessions import list_sessions

warnings.filterwarnings("ignore", category=RuntimeWarning)

REST_WINDOW_S = (-0.05, -0.003)  # ボール静止位置を取る区間(インパクト前)
ARC_WINDOW_S = (-0.2, 0.0)  # 足先の最下点を探す区間
HYPOTHESIS = ("com_minus_ball_x_m", "trunk_lean_fwd_deg", "support_ankle_minus_ball_x_m", "arc_phase_ms",
              "contact_dy_m", "foot_pitch_deg", "toe_vel_elev_deg",  # 重心が後ろ/接触位置/スイング位相の仮説に直結する要因(2D)
              "xs_swing_axis_tilt_deg", "xs_kick_shank_lat_deg", "xs_kick_thigh_lat_deg", "xs_pelvis_lean_kickside_deg",
              "xs_pelvis_lean_fwd_deg", "xs_support_shank_lat_deg", "xs_support_shank_fwd_deg")  # 横方向の要因(Xsens)


def _pvalue(a: pd.Series, b: pd.Series) -> float:
    from scipy.stats import spearmanr

    ok = a.notna() & b.notna()
    return float(spearmanr(a[ok], b[ok])[1]) if ok.sum() >= 4 else float("nan")


SKIP_CORR = ("land_y_m", "land_z_m", "land_y_kickside_m", "land_major_m", "land_diag_m", "trial_no", "impact_frame", "ball_x_m", "ball_y_m",
             "xs_impact_t_s", "xs_kickleg_ratio")  # 要因ではない列(Xsensのインパクト時刻・センサー対応の確認値など)


def _deg(dx: float, dy: float) -> float:
    return float(np.degrees(np.arctan2(dy, dx)))


def ball_rest(k: pd.DataFrame, sgn: float) -> np.ndarray:
    """インパクト前のボール静止位置(中央値) [x, y] m。取れなければ NaN。"""
    if "ball_x" not in k:
        return np.full(2, np.nan)
    t = k["t_rel_impact_s"].to_numpy()
    sel = (t >= REST_WINDOW_S[0]) & (t <= REST_WINDOW_S[1])
    return np.array([sgn * np.nanmedian(k.loc[sel, "ball_x"]), np.nanmedian(k.loc[sel, "ball_y"])])


def ball_launch(k: pd.DataFrame, sgn: float) -> tuple[float, float]:
    """インパクト後のボール速度の(仰角 deg, 速さ m/s)。矢状面(2D)内のみ。"""
    if "ball_x" not in k:
        return np.nan, np.nan
    t = k["t_rel_impact_s"].to_numpy()
    sel = (t >= BALL_WINDOW_S[0] - 1e-9) & (t <= BALL_WINDOW_S[1] + 1e-9)
    p = k.loc[sel, ["ball_x", "ball_y"]].to_numpy()
    ok = np.isfinite(p).all(1)
    if ok.sum() < 3:
        return np.nan, np.nan
    vx, vy = (np.polyfit(t[sel][ok], p[ok, i], 1)[0] for i in range(2))
    return _deg(sgn * vx, vy), float(np.hypot(vx, vy))


def trial_factors(k: pd.DataFrame, info: dict) -> dict:
    side = info["kick_side"]
    sup = "right" if side == "left" else "left"
    sgn = kick_sign(k, side)
    metrics, ts = analyze_trial(k, info)
    t = k["t_rel_impact_s"].to_numpy()
    i1 = int(np.nanargmin(np.abs(t)))
    P = lambda n: _p(k, n, sgn)  # noqa: E731

    ball = ball_rest(k, sgn)
    toe, heel, ank, knee, hip = (P(f"{side}_{n}") for n in ("big_toe", "heel", "ankle", "knee", "hip"))
    s_ank, s_heel, s_toe = (P(f"{sup}_{n}") for n in ("ankle", "heel", "big_toe"))
    pelvis, neck = P("pelvis"), P("neck")
    com = np.column_stack([ts["body_com_x"], ts["body_com_y"]])
    foot_mid = 0.5 * (heel + toe)

    # 足先速度の向き(インパクト直前5ms平均)
    sel = (t >= -0.005) & (t <= 0.0)
    v = np.array([sgn * np.nanmean(k.loc[sel, f"{side}_big_toe_vx"]), np.nanmean(k.loc[sel, f"{side}_big_toe_vy"])])

    # スイング円弧上の接触位相: 足先の最下点からインパクトまでの時間・距離
    arc = (t >= ARC_WINDOW_S[0]) & (t <= ARC_WINDOW_S[1])
    j = int(np.flatnonzero(arc)[np.nanargmin(toe[arc, 1])])
    launch_deg, launch_speed = ball_launch(k, sgn)

    out = dict(
        impact_frame=info.get("impact_frame"), kick_side=side,
        com_minus_ball_x_m=com[i1, 0] - ball[0],  # 負=重心がボールより後ろ
        com_height_m=com[i1, 1],
        hip_minus_ball_x_m=hip[i1, 0] - ball[0],
        support_ankle_minus_ball_x_m=s_ank[i1, 0] - ball[0],
        support_toe_minus_ball_x_m=s_toe[i1, 0] - ball[0],
        trunk_lean_fwd_deg=_deg(neck[i1, 1] - pelvis[i1, 1], neck[i1, 0] - pelvis[i1, 0]),  # 鉛直から前方へ(前傾が+)
        thigh_fwd_deg=_deg(-(knee[i1, 1] - hip[i1, 1]), knee[i1, 0] - hip[i1, 0]),  # 鉛直下向きから前方へ
        shank_fwd_deg=_deg(-(ank[i1, 1] - knee[i1, 1]), ank[i1, 0] - knee[i1, 0]),
        foot_pitch_deg=_deg(toe[i1, 0] - heel[i1, 0], toe[i1, 1] - heel[i1, 1]),  # つま先上が+
        contact_dx_m=foot_mid[i1, 0] - ball[0],  # 足の中点-ボール中心
        contact_dy_m=foot_mid[i1, 1] - ball[1],  # +=足がボール中心より上
        toe_dx_m=toe[i1, 0] - ball[0],
        toe_dy_m=toe[i1, 1] - ball[1],
        toe_vel_elev_deg=_deg(v[0], v[1]),  # インパクト直前の足先速度の仰角
        toe_vel_mps=float(np.hypot(*v)),
        arc_lowest_t_ms=t[j] * 1000,  # 足先最下点の時刻(インパクト基準)
        arc_phase_ms=-t[j] * 1000,  # 最下点→インパクトの時間(+=最下点より後で接触=上昇中)
        arc_phase_dx_m=toe[i1, 0] - toe[j, 0],
        arc_phase_rise_m=toe[i1, 1] - toe[j, 1],
        ball_launch_elev_deg=launch_deg, ball_launch_speed_mps=launch_speed,
        ball_x_m=ball[0], ball_y_m=ball[1],
    )
    for key in ("ball_speed_mps", "toe_speed_impact_mps", "m_eff_impact_kg", "knee_flexion_impact_deg", "notes"):
        out[key] = metrics.get(key)
    out["launch_minus_toevel_deg"] = launch_deg - out["toe_vel_elev_deg"]
    return out


def load_landing() -> pd.DataFrame:
    p = ROOT / "config" / "landing.csv"
    if not p.exists():
        return pd.DataFrame(columns=["session", "trial", "y_m", "z_m"])
    return pd.read_csv(p, encoding="utf-8-sig", comment="#")


def tilt_deg(y: np.ndarray, z: np.ndarray) -> float:
    """(y,z)分布の長軸が y 軸となす角(度, -90〜90)。+=y が大きいほど z も大きい。"""
    c = np.cov(np.vstack([y, z]))
    w, V = np.linalg.eigh(c)
    major = V[:, np.argmax(w)]
    a = np.degrees(np.arctan2(major[1], major[0]))
    return float((a + 90) % 180 - 90)


def run_factors(sessions: list[str] | None = None, root: Path | None = None, log=print, xsens: bool = True) -> pd.DataFrame:
    root = root or ROOT / "output"
    rows = []
    for sess in sessions or list_sessions():
        for kin in sorted((root / sess).glob("*/kinematics.csv")):
            sj = kin.parent / "summary.json"
            if not sj.exists():
                continue
            info = json.loads(sj.read_text(encoding="utf-8"))
            k = pd.read_csv(kin)
            if "t_rel_impact_s" not in k:
                log(f"[スキップ] {kin.parent.name}: インパクト未設定")
                continue
            try:
                f = trial_factors(k, info)
            except Exception as e:  # 1試行の失敗で全体を止めない
                log(f"[スキップ] {kin.parent.name}: {type(e).__name__}: {e}")
                continue
            m = re.search(r"(\d+)$", kin.parent.name)
            rows.append(dict(session=sess, trial=kin.parent.name, trial_no=int(m[1]) if m else None, **f))
    df = pd.DataFrame(rows)
    if df.empty:
        log("対象試行がありません")
        return df
    if xsens:  # Xsens(IMU)の3D姿勢・スイング面の要因。試行番号(trial_no = taskN)で2Dと結合する
        from .xsensfactors import xsens_factors

        parts = []
        for sess, g in df.groupby("session"):
            x = xsens_factors(sess, str(g["kick_side"].iloc[0]), log=log)
            log(f"[{sess}] Xsens {len(x)} 試行(蹴りを検出できた試行)" if len(x) else f"[{sess}] Xsens 使える試行なし(data/xsens/{sess}/ の有無・センサー対応を確認)")
            if len(x):
                parts.append(x)
        if parts:
            df = df.merge(pd.concat(parts, ignore_index=True), on=["session", "trial_no"], how="left")
    land = load_landing()
    if len(land):
        for sess in dict.fromkeys([*df["session"], *land["session"]]):  # 試行名の食い違いを見つけるための突き合わせ状況
            proc = set(df.loc[df["session"] == sess, "trial"])
            lan = set(land.loc[land["session"] == sess, "trial"])
            log(f"[{sess}] 処理済み {len(proc)} / 着弾 {len(lan)} / 一致 {len(proc & lan)}"
                + (f"  着弾のみ(動画・出力なし): {sorted(lan - proc)}" if lan - proc else "")
                + (f"  処理のみ(着弾なし): {sorted(proc - lan)}" if proc - lan and lan else ""))
        df = df.merge(land[["session", "trial", "y_m", "z_m"]].rename(columns={"y_m": "land_y_m", "z_m": "land_z_m"}),
                      on=["session", "trial"], how="left")
        right = (df["kick_side"] == "right").map({True: 1.0, False: -1.0})
        df["land_y_kickside_m"] = df["land_y_m"] * right  # 蹴り足側が+
        df["land_major_m"] = np.nan  # 被験者ごとの着弾分布の長軸(蹴り足側y,z)上の位置(平均0)。+=蹴り足側・上方向
        df["land_diag_m"] = np.nan  # 全被験者で共通の斜め方向((y,z)=(1,1)/√2)上の位置(被験者ごとに平均0)。+=蹴り足側・上
        for _, g in df.dropna(subset=["land_z_m"]).groupby("session"):
            if len(g) >= 3:
                y, z = g["land_y_kickside_m"].to_numpy(), g["land_z_m"].to_numpy()
                w, V = np.linalg.eigh(np.cov(np.vstack([y, z])))
                v = V[:, np.argmax(w)]
                v = -v if v[0] < 0 else v
                df.loc[g.index, "land_major_m"] = (y - y.mean()) * v[0] + (z - z.mean()) * v[1]
                df.loc[g.index, "land_diag_m"] = ((y - y.mean()) + (z - z.mean())) / np.sqrt(2)
    out = root / "analysis" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "kick_factors.csv", index=False, float_format="%.4f")
    log(f"保存: {out / 'kick_factors.csv'} ({len(df)} 試行)")

    if "land_z_m" in df and df["land_z_m"].notna().sum() >= 5:
        corr = []
        for sess, g in df.dropna(subset=["land_z_m"]).groupby("session"):
            if len(g) < 5:
                continue
            log(f"[{sess}] 着弾分布の長軸の傾き(蹴り足側y基準): {tilt_deg(g['land_y_kickside_m'].to_numpy(), g['land_z_m'].to_numpy()):.1f}°  (n={len(g)})")
            for c in g.select_dtypes("number").columns:
                if c in SKIP_CORR or g[c].nunique() < 3:
                    continue
                corr.append(_corr_row(sess, c, g, g))
        sess_ok = [s for s, g in df.dropna(subset=["land_z_m"]).groupby("session") if len(g) >= 5]
        exp = pd.to_numeric(land.drop_duplicates("session").set_index("session")["experience_y"], errors="coerce") if "experience_y" in land else pd.Series(dtype=float)
        groups = {"ALL(within-subject z)": sess_ok, "EXP(>=8y, within-subject z)": [s for s in sess_ok if exp.get(s, np.nan) >= 8]}
        for label, ss in groups.items():  # 被験者内で標準化して合算(被験者の体格差・癖を除く)
            if len(ss) >= 2:
                corr += _pooled_rows(df[df["session"].isin(ss)], label)
        if corr:
            cdf = pd.DataFrame(corr)
            cdf.to_csv(out / "kick_factors_corr.csv", index=False, float_format="%.3f")
            log(f"保存: {out / 'kick_factors_corr.csv'}")
            for sess, g in cdf.groupby("session", sort=False):  # 長軸上の位置と相関の強い要因(上位6。* p<0.05 は要因が約50個あるので偶然も含む)
                log(f"[{sess}] 長軸位置(+=蹴り足側・上)と相関の強い要因: {_top(g, 'major')}")
                hy = g[g["factor"].isin(HYPOTHESIS)].set_index("factor")
                log(f"[{sess}]   仮説の要因: " + ", ".join(f"{k} {hy.loc[k, 'rho_vs_major']:+.2f}{'*' if hy.loc[k, 'p_vs_major'] < 0.05 else ''}" for k in HYPOTHESIS if k in hy.index))
                if sess.startswith(("ALL", "EXP")):
                    for key, name in (("diag", "斜め方向(蹴り足側・上)"), ("z", "上下z"), ("kickside_y", "蹴り足側y")):
                        log(f"[{sess}]   {name}と相関の強い要因: {_top(g, key)}")
                if sess.startswith("ALL"):  # 合算で強い要因が被験者ごとにも同じ向きか(符号の一貫性)
                    per = cdf[~cdf["session"].str.startswith(("ALL", "EXP"))]
                    for key, name in (("z", "上下z"), ("diag", "斜め"), ("kickside_y", "蹴り足側y")):
                        for r in g.reindex(g[f"rho_vs_{key}"].abs().sort_values(ascending=False).index).head(3).itertuples():
                            v = per[per["factor"] == r.factor].set_index("session")[f"rho_vs_{key}"]
                            log(f"[{sess}]   {name}の上位 {r.factor}: 全体 {getattr(r, 'rho_vs_' + key):+.2f} | 被験者別 "
                                + ", ".join(f"{s} {x:+.2f}" for s, x in v.items()))
        for sess, g in df.dropna(subset=["land_z_m"]).groupby("session"):
            if len(g) >= 3:
                path = root / "analysis" / "landing" / f"{sess}_factors.png"
                try:
                    plot_factor_scatter(g, sess, path)
                    log(f"保存: {path}")
                except OSError as e:  # 図が保存できなくても表(csv)は出ているので続行する
                    log(f"[警告] 図を保存できません {path}: {e}")
        swing_plane_check(df, root, log)
    return df


def swing_plane_check(df: pd.DataFrame, root: Path, log=print) -> pd.DataFrame:
    """スイング面モデルの予測と、実測の着弾の「左右-上下の結びつき」を比べる。

    蹴り脚を、股関節まわりの1本の回転軸(Xsensで測ったスイング面の傾き τ: 蹴り足側の端が水平より上=+)で振る剛体回転とみなすと、
    足先の速度は軸に垂直な面内にあり、前方 f と m = cosτ·上 − sinτ·蹴り足側 で張られる。速度が前方から角度 ψ だけ上を向くとき、
      上下成分 ≈ ψ·cosτ、 蹴り足側成分 ≈ −ψ·sinτ  →  蹴り足側への偏り y ≈ −tanτ · (上下の偏り z)
    つまり面が傾いていると「高く飛ぶほど蹴り足の反対側へ逸れる」(傾き y/z = −tanτ)。着弾が実測で y/z > 0(蹴り足側へ逸れるほど高い)なら、
    この単純な面内の速度方向では説明できず、接触点・足面の向きなど別の要因が効いていることを示す(ボールは足の速度方向ではなく接触面の法線方向へ飛ぶ)。
    出力: 被験者ごとの予測の傾き -tanτ̄ と実測の回帰の傾き(標準誤差つき)。"""
    from scipy.stats import linregress

    need = {"xs_swing_axis_tilt_deg", "land_z_m", "land_y_kickside_m"}
    if not need <= set(df.columns):
        return pd.DataFrame()
    rows = []
    for sess, g in df.dropna(subset=list(need)).groupby("session"):
        if len(g) < 4:
            continue
        tau = np.radians(g["xs_swing_axis_tilt_deg"].mean())
        pred = -np.tan(tau)
        res = linregress(g["land_z_m"], g["land_y_kickside_m"])  # y = β z + c
        diff = (res.slope - pred) / res.stderr if res.stderr > 0 else np.nan
        rows.append(dict(session=sess, n=len(g), tau_deg=np.degrees(tau), tau_sd_deg=g["xs_swing_axis_tilt_deg"].std(),
                         beta_pred=pred, beta_obs=res.slope, beta_obs_se=res.stderr, diff_in_se=diff, r_obs=res.rvalue))
        log(f"[{sess}] スイング面モデル(n={len(g)}): 平均傾き τ={np.degrees(tau):+.0f}° → 予測 y={pred:+.2f}·z (高いほど蹴り足の反対側) / "
            f"実測 y={res.slope:+.2f}±{res.stderr:.2f}·z (r={res.rvalue:+.2f}) → 差 {diff:+.1f} SE")
    out = pd.DataFrame(rows)
    if not out.empty:
        p = root / "analysis" / "tables" / "swing_plane_check.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(p, index=False, float_format="%.3f")
        log(f"保存: {p}")
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, len(out), figsize=(3.6 * len(out), 3.6), squeeze=False)
            for ax, r in zip(axes[0], out.itertuples()):
                g = df[df["session"] == r.session].dropna(subset=list(need))
                ax.scatter(g["land_y_kickside_m"], g["land_z_m"], s=16, color="#2a78d6")
                zc, yc = g["land_z_m"].mean(), g["land_y_kickside_m"].mean()
                zz = np.array([g["land_z_m"].min(), g["land_z_m"].max()])
                ax.plot(yc + r.beta_obs * (zz - zc), zz, color="#2a78d6", label="observed")
                ax.plot(yc + r.beta_pred * (zz - zc), zz, color="#e0301e", ls="--", label=f"swing-plane model (tilt {r.tau_deg:+.0f} deg)")
                ax.set_title(r.session, fontsize=9)
                ax.set_xlabel("landing y, kick side + (m)", fontsize=8)
                ax.set_ylabel("landing z (m)", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.25)
            axes[0][0].legend(fontsize=6)
            fig.tight_layout()
            save_figure(fig, root / "analysis" / "landing" / "swing_plane_check.png", dpi=130)
            plt.close(fig)
        except OSError as e:
            log(f"[警告] 図を保存できません: {e}")
    return out


LAND_KEYS = (("kickside_y", "land_y_kickside_m"), ("z", "land_z_m"), ("major", "land_major_m"), ("diag", "land_diag_m"))


def _corr_row(label: str, c: str, g: pd.DataFrame, zs: pd.DataFrame) -> dict:
    row = dict(session=label, factor=c, n=int(len(zs)))
    for key, col in LAND_KEYS:
        row[f"rho_vs_{key}"] = zs[c].corr(zs[col], method="spearman")
        row[f"p_vs_{key}"] = _pvalue(zs[c], zs[col])
    return row


def _pooled_rows(pooled: pd.DataFrame, label: str) -> list[dict]:
    rows = []
    for c in pooled.select_dtypes("number").columns:
        if c in SKIP_CORR:
            continue
        zs = pooled.groupby("session")[[c, *[col for _, col in LAND_KEYS]]].transform(lambda s: (s - s.mean()) / (s.std() or np.nan))
        ok = zs.notna().all(axis=1)
        if ok.sum() >= 10:
            rows.append(_corr_row(label, c, pooled, zs[ok]))
    return rows


def _top(g: pd.DataFrame, key: str, k: int = 6) -> str:
    top = g.reindex(g[f"rho_vs_{key}"].abs().sort_values(ascending=False).index).head(k)
    return ", ".join(f"{r.factor} {getattr(r, 'rho_vs_' + key):+.2f}{'*' if getattr(r, 'p_vs_' + key) < 0.05 else ''}" for r in top.itertuples())


SCATTER_FACTORS = [
    ("com_minus_ball_x_m", "COM - ball x (m)"), ("support_ankle_minus_ball_x_m", "support ankle - ball x (m)"),
    ("trunk_lean_fwd_deg", "trunk lean fwd (deg)"), ("foot_pitch_deg", "foot pitch (deg)"),
    ("toe_vel_elev_deg", "toe velocity elevation (deg)"), ("arc_phase_ms", "swing-arc phase (ms after lowest)"),
    ("contact_dy_m", "foot mid - ball center y (m)"), ("ball_launch_elev_deg", "ball launch elevation 2D (deg)"),
]


def plot_factor_scatter(g: pd.DataFrame, sess: str, out: Path) -> None:
    """各要因(行) × 着弾の z / 蹴り足側 y (列) の散布図。点は試行番号、各図にSpearmanのrhoを表示。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(SCATTER_FACTORS), 2, figsize=(8, 2.3 * len(SCATTER_FACTORS)))
    for r, (col, label) in enumerate(SCATTER_FACTORS):
        for c, (lcol, llabel) in enumerate((("land_z_m", "landing z (m)"), ("land_y_kickside_m", "landing y, kick side + (m)"))):
            ax = axes[r, c]
            ok = g[[col, lcol]].notna().all(axis=1)
            x, y = g.loc[ok, lcol], g.loc[ok, col]
            ax.scatter(x, y, s=14, color="#2a78d6")
            for xi, yi, no in zip(x, y, g.loc[ok, "trial_no"]):
                ax.annotate(str(no), (xi, yi), fontsize=7, xytext=(3, 3), textcoords="offset points")
            rho = x.corr(y, method="spearman") if ok.sum() >= 4 else np.nan
            ax.set_title(f"rho={rho:+.2f} (n={ok.sum()})", fontsize=8, loc="right")
            ax.set_ylabel(label, fontsize=7)
            ax.set_xlabel(llabel, fontsize=7) if r == len(SCATTER_FACTORS) - 1 else None
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25)
    fig.suptitle(f"{sess}: impact-time factors vs landing", fontsize=10)
    fig.tight_layout()
    save_figure(fig, out, dpi=130)
    plt.close(fig)
