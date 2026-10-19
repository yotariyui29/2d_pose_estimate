"""着弾楕円の傾きは、接触モデルの感度だけで生じるのか、実行変数間の共変動が必要なのか。

背景: 経験者の着弾は「蹴り足側へそれると上方」の斜めの楕円になる。Müller & Sternad(2004)/Cohen & Sternad(2009)のTNC分解では、
結果のばらつきは ①許容度(結果が実行誤差にどれだけ敏感か=感度行列) ②ノイズ ③共変動 に分かれる。ここでは contact_sensitivity.py の
前向きモデル(剛体衝突+飛翔)の感度行列 J を①として使い、
  (a) J(着弾 y,z / 実行変数)に非対角(傾きを作る結合)があるか、実行変数が独立にばらつくだけで着弾の傾きが出るか(基準点を振って頑健性も見る)
  (b) 実測の着弾の相関・傾き・軸比とそのブートストラップ95%信頼区間(n=7〜11でどこまで言えるか)
を出す。注意: 感度は仮定した基準点・典型SD(contact_sensitivity.INPUTS)と接触モデルの仮定(足の曲率半径5cm等)に依存する。
左右の実行変数(方位・接触点の左右)は側面1台の2Dでは測れないので、必要な共変動の中身(どの実行変数の組か)は特定できない。

実行: PYTHONPATH=src python -m lit_followup.tilt_from_model
出力: output/lit_followup/04_tilt_from_model/{jacobian_base.csv, jacobian_sweep.csv, tilt_observed.csv, tilt_from_model.png(+_ja)}
"""
from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy import stats

from pose2d.contact_sensitivity import OUTLIER_ELEV_DEG, Params, sensitivities
from pose2d.figio import save_figure

from ._common import OUT_DIR, tables_dir

SUBDIR = "04_tilt_from_model"
N_BOOT = 5000
CHI2_95 = 5.991  # 2自由度の95%


def independent_cov(sens: pd.DataFrame) -> np.ndarray:
    """実行変数が互いに独立に典型SDでばらつくときの着弾(y蹴り足側, z)の共分散 = Σ_i g_i g_iᵀ (g_i: 入力i の+1SDあたりの着弾の変化)。"""
    g = sens[["dy_k_m_per_sd", "dz_m_per_sd"]].to_numpy()
    return g.T @ g


def _corr(S: np.ndarray) -> float:
    return float(S[0, 1] / np.sqrt(S[0, 0] * S[1, 1]))


def tilt_deg(S: np.ndarray) -> float:
    w, V = np.linalg.eigh(S)
    v = V[:, np.argmax(w)]
    return float((np.degrees(np.arctan2(v[1], v[0])) + 90) % 180 - 90)


def jacobian_sweep() -> tuple[pd.DataFrame, pd.DataFrame]:
    base = Params()
    sens0 = sensitivities(base)
    sweeps = {"speed": (16.0, 20.0, 24.0), "elev": (0.0, 3.0, 8.0), "M": (1.0, 1.4, 2.0), "e": (0.45, 0.6, 0.7), "mu": (0.2, 0.4, 0.6)}
    rows = []
    for name, vals in sweeps.items():
        for v in vals:
            s = sensitivities(replace(base, **{name: v}))
            S = independent_cov(s)
            rows.append(dict(varied=name, value=v, aimed_h_up_cm=100 * s.attrs["base"].h_up, ball_speed_mps=s.attrs["ball_speed"],
                             spin_rpm=s.attrs["spin_rpm"], sd_y_ind_m=np.sqrt(S[0, 0]), sd_z_ind_m=np.sqrt(S[1, 1]), corr_yz_ind=_corr(S)))
    return sens0, pd.DataFrame(rows)


def _circ_ci(angles: np.ndarray) -> tuple[float, float, float]:
    """傾き(180°周期)の円周平均と、それに対する偏差のパーセンタイルによる95%区間。"""
    c = 0.5 * np.degrees(np.arctan2(np.mean(np.sin(np.radians(2 * angles))), np.mean(np.cos(np.radians(2 * angles)))))
    dev = (angles - c + 90) % 180 - 90
    lo, hi = np.percentile(dev, [2.5, 97.5])
    return float(c), float(c + lo), float(c + hi)


def observed(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows, data = [], {}
    for sess, g in df.dropna(subset=["land_z_m", "land_y_kickside_m", "toe_vel_elev_deg"]).groupby("session"):
        if len(g) < 5:
            continue
        g = g[(g["toe_vel_elev_deg"] - g["toe_vel_elev_deg"].median()).abs() <= OUTLIER_ELEV_DEG]  # contact_sensitivity.decompose と同じ除外
        if len(g) < 5:
            continue
        y, z = g["land_y_kickside_m"].to_numpy(), g["land_z_m"].to_numpy()
        S = np.cov(np.vstack([y, z]))
        w = np.sort(np.linalg.eigvalsh(S))[::-1]
        bt, br = [], []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(y), len(y))
            if np.ptp(y[i]) == 0 or np.ptp(z[i]) == 0:
                continue
            Sb = np.cov(np.vstack([y[i], z[i]]))
            bt.append(tilt_deg(Sb))
            br.append(_corr(Sb))
        bt, br = np.array(bt), np.array(br)
        c, lo, hi = _circ_ci(bt)
        rows.append(dict(session=sess, n=len(y), sd_y_m=np.sqrt(S[0, 0]), sd_z_m=np.sqrt(S[1, 1]), tilt_deg=tilt_deg(S),
                         tilt_boot_ci_lo=lo, tilt_boot_ci_hi=hi, axis_ratio=float(np.sqrt(w[0] / w[1])), pearson_r=_corr(S),
                         r_boot_ci_lo=float(np.percentile(br, 2.5)), r_boot_ci_hi=float(np.percentile(br, 97.5)),
                         pearson_p=float(stats.pearsonr(y, z)[1]), spearman_rho=float(stats.spearmanr(y, z)[0]), spearman_p=float(stats.spearmanr(y, z)[1]),
                         cov_share_of_variance=float(2 * abs(S[0, 1]) / (S[0, 0] + S[1, 1]))))
        data[sess] = (y, z, bt)
    out = pd.DataFrame(rows)
    out.attrs["data"] = data
    return out


def _ellipse(ax, mean, S, **kw) -> None:
    w, V = np.linalg.eigh(S)
    ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    ax.add_patch(Ellipse(mean, 2 * np.sqrt(CHI2_95 * w[1]), 2 * np.sqrt(CHI2_95 * w[0]), angle=ang, fill=False, **kw))


def build_figure(obs: pd.DataFrame) -> plt.Figure:
    data = obs.attrs["data"]
    fig, axes = plt.subplots(2, len(obs), figsize=(3.6 * len(obs), 7), constrained_layout=True, squeeze=False)
    for j, r in enumerate(obs.itertuples()):
        y, z, bt = data[r.session]
        ax = axes[0, j]
        ax.scatter(y, z, s=22, color="#0072b2", zorder=3)
        m = np.array([y.mean(), z.mean()])
        _ellipse(ax, m, np.cov(np.vstack([y, z])), color="#0072b2", lw=1.6, label="observed 95% ellipse")
        _ellipse(ax, m, np.diag([r.sd_y_m ** 2, r.sd_z_m ** 2]), color="#d55e00", lw=1.4, ls="--", label="same SD, no covariation")
        ax.set_aspect("equal", adjustable="datalim")
        ax.autoscale()
        ax.set_title(f"{r.session} (n={r.n}): r={r.pearson_r:+.2f} [{r.r_boot_ci_lo:+.2f}, {r.r_boot_ci_hi:+.2f}], tilt {r.tilt_deg:+.0f} deg, axis ratio {r.axis_ratio:.1f}", fontsize=7)
        ax.set_xlabel("Landing y, kick side + (m)", fontsize=8)
        ax.set_ylabel("Landing z (m)", fontsize=8)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
        axb = axes[1, j]
        axb.hist(bt, bins=36, range=(-90, 90), color="#999999")
        axb.axvline(r.tilt_deg, color="#d55e00")
        axb.set_xlabel("bootstrap tilt of the long axis (deg)", fontsize=8)
        axb.set_ylabel("count", fontsize=8)
        axb.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=6)
    return fig


if __name__ == "__main__":
    out = OUT_DIR / SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    sens0, sweep = jacobian_sweep()
    cols = ["input", "label", "unit", "typical_sd", "dy_k_m_per_step", "dz_m_per_step", "dy_k_m_per_sd", "dz_m_per_sd"]
    sens0[cols].to_csv(out / "jacobian_base.csv", index=False, float_format="%.4f")
    sweep.to_csv(out / "jacobian_sweep.csv", index=False, float_format="%.4f")
    S0 = independent_cov(sens0)
    print("感度行列(基準点, +1典型SDあたり):")
    print(sens0[["input", "dy_k_m_per_sd", "dz_m_per_sd"]].round(3).to_string(index=False))
    print(f"独立ノイズだけの着弾: SD y {np.sqrt(S0[0, 0]):.2f} m, SD z {np.sqrt(S0[1, 1]):.2f} m, 相関 {_corr(S0):+.3f}")
    print(f"基準点を振っても相関は {sweep.corr_yz_ind.min():+.3f}〜{sweep.corr_yz_ind.max():+.3f}")
    obs = observed(pd.read_csv(tables_dir() / "kick_factors.csv"), np.random.default_rng(0))
    obs.drop(columns=[], errors="ignore").round(3).to_csv(out / "tilt_observed.csv", index=False)
    print(obs.round(2).to_string(index=False))
    fig = build_figure(obs)
    save_figure(fig, out / "tilt_from_model.png", dpi=140)
    plt.close(fig)
