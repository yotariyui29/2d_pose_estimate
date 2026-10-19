"""imu_video_3d の3D運動学を OpenSim で再生できる形式（.osim/.mot/.trc + Geometry）に書き出す。

  - モデル: Pose2Sim の Model_Pose2Sim_simple.osim を、2Dから求めたセグメント長(大腿・下腿・足部)と股幅に合わせてスケール
            （座標の clamp は解除。体幹・腕は直立のまま＝計測なし）
  - .mot : 骨盤(傾き3+並進3)・両脚の股(3)・膝・足首の座標。Xsens姿勢から、モデルの関節軸に合わせた順運動学の逆（軸分解）で算出
  - .trc : 股・膝・足首・つま先のFK位置（マーカー表示用）
  - OpenSim座標系: x=正面(直立時), y=鉛直上, z=被験者の右。時刻 = インパクト基準の時刻+0.9（0.9s=インパクト）
GUI: File > Open Model で .osim → File > Load Motion で .mot。
Usage: python src/export_opensim.py --trial kurogi_t1 [--source auto|pose2d|kinovea]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import opensim as osim
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from .imu_video_3d import FS, Z, run
from .trials import ROOT, Trial, load_trials

MODEL_SRC = ROOT / "assets" / "Model_Pose2Sim_simple.osim"  # Pose2Sim同梱モデルのコピー
IMPACT_T = 0.9  # mot時刻 = インパクト基準の時刻 + 0.9


def model_meshes() -> list[str]:
    """モデル(.osim)が参照するメッシュのファイル名。"""
    import re

    return sorted(set(re.findall(r"<mesh_file>([^<]+)</mesh_file>", MODEL_SRC.read_text(encoding="utf-8"))))


def geometry_dirs() -> list[Path]:
    """メッシュ(Geometry)の候補フォルダ(優先順): paths.toml の geometry_dir / 環境変数 OPENSIM_GEOMETRY / Pose2Sim の同梱物 /
    OpenSim 本体のインストール先(`<OpenSim x.y>/Geometry`)。"""
    import os

    out: list[Path] = []
    try:
        from .trials import load_paths

        if "geometry_dir" in load_paths():
            out.append(load_paths()["geometry_dir"])
    except Exception:  # noqa: BLE001
        pass
    if os.environ.get("OPENSIM_GEOMETRY"):
        out.append(Path(os.environ["OPENSIM_GEOMETRY"]))
    try:
        import Pose2Sim

        out.append(Path(Pose2Sim.__file__).parent / "OpenSim_Setup" / "Geometry")
    except ImportError:
        pass
    for pat in ("F:/yamada/OpenSim */Geometry", "C:/OpenSim */Geometry", "C:/Program Files/OpenSim */Geometry",
                "/mnt/f/yamada/OpenSim */Geometry", "/mnt/c/OpenSim */Geometry"):
        base, _, rest = pat.partition("*")
        root = Path(base).parent
        if root.is_dir():
            out += sorted(root.glob(Path(base).name + "*/Geometry"), reverse=True)
    return [d for d in out if d.is_dir()]


def copy_geometry(dst: Path) -> bool:
    """モデルが使うメッシュだけを、候補フォルダから dst にコピーする。足りないメッシュは警告する。全部そろえば True。"""
    need = model_meshes()
    dirs = geometry_dirs()
    if not dirs:
        return False
    dst.mkdir(parents=True, exist_ok=True)
    missing = []
    for name in need:
        src = next((d / name for d in dirs if (d / name).exists()), None)
        if src is None:
            missing.append(name)
        elif not (dst / name).exists():
            shutil.copy2(src, dst / name)
    if missing:
        print(f"警告: メッシュ {len(missing)} 個が見つかりません(例: {missing[:3]}): 探した場所 {[str(d) for d in dirs]}")
    return not missing


def v3(a) -> osim.Vec3:
    return osim.Vec3(*[float(x) for x in a])


def body_p(m: osim.Model, st, b: str) -> np.ndarray:
    return m.getBodySet().get(b).getTransformInGround(st).p().to_numpy()


def body_R(m: osim.Model, st, b: str) -> R:
    M = m.getBodySet().get(b).getTransformInGround(st).R()
    return R.from_matrix(np.array([[M.get(i, j) for j in range(3)] for i in range(3)]))


def make_scaled_model(L: dict[str, float], hip_width: float) -> osim.Model:
    m = osim.Model(str(MODEL_SRC))
    st = m.initSystem()
    m.realizePosition(st)
    P = lambda b: body_p(m, st, b)
    f_thigh = L["thigh"] / np.linalg.norm(P("femur_l") - P("tibia_l"))
    f_shank = L["shank"] / np.linalg.norm(P("tibia_l") - P("talus_l"))
    f_foot = L["foot"] / np.linalg.norm(P("talus_l") - P("toes_l"))
    f_pelv = hip_width / abs(P("femur_r")[2] - P("femur_l")[2])
    factors = {"pelvis": (1, 1, f_pelv)}
    for s in "rl":
        factors.update({f"femur_{s}": [f_thigh] * 3, f"tibia_{s}": [f_shank] * 3,
                        f"talus_{s}": [f_foot] * 3, f"calcn_{s}": [f_foot] * 3, f"toes_{s}": [f_foot] * 3})
    ss = osim.ScaleSet()
    for name, f in factors.items():
        sc = osim.Scale()
        sc.setSegmentName(name)
        sc.setScaleFactors(v3(f))
        sc.setApply(True)
        ss.cloneAndAppend(sc)
    m.scale(st, ss, True)
    cs = m.getCoordinateSet()
    for i in range(cs.getSize()):
        cs.get(i).set_clamped(False)
    print(f"scale: thigh {f_thigh:.3f} shank {f_shank:.3f} foot {f_foot:.3f} pelvis(z) {f_pelv:.3f}")
    return m


def effective_axis(m: osim.Model, coord: str, body: str) -> np.ndarray:
    """零点姿勢(全体が地面と同じ向き)で座標を微小回転させたときの、bodyの地面系回転軸。"""
    st = m.initSystem()
    c = m.getCoordinateSet().get(coord)
    c.setValue(st, 0.05)
    m.realizePosition(st)
    rv = body_R(m, st, body).as_rotvec()
    return rv / np.linalg.norm(rv)


def chain(axes: list[np.ndarray], q: np.ndarray, reverse: bool) -> R:
    rots = [R.from_rotvec(a * x) for a, x in zip(axes, q)]
    out = R.identity()
    for r in (rots[::-1] if reverse else rots):
        out = out * r
    return out


def decompose(target: list[R], axes: list[np.ndarray], reverse: bool) -> np.ndarray:
    """3軸の回転の積が target に一致する角度列(rad)（前フレームから初期化）。"""
    q, prev = np.zeros((len(target), 3)), np.zeros(3)
    for i, t in enumerate(target):
        prev = least_squares(lambda x: (chain(axes, x, reverse).inv() * t).as_rotvec(), prev, xtol=1e-10, ftol=1e-10).x
        q[i] = prev
    return q


def hinge_dir(parent: list[R], child: list[R], axis: np.ndarray, v0: np.ndarray) -> np.ndarray:
    """1自由度関節の角度: 子セグメントの長軸(零点でv0)の向きが目標に最も合う、axis まわりの角度(rad)。"""
    out = []
    flat = lambda v: v - (v @ axis) * axis
    for p, c in zip(parent, child):
        a, b = flat(v0), flat(p.inv().apply(c.apply(v0)))
        out.append(np.arctan2(axis @ np.cross(a, b), a @ b))
    return np.array(out)


def write_mot(path: Path, names: list[str], time: np.ndarray, data: np.ndarray, title: str) -> None:
    with open(path, "w") as f:
        f.write(f"{title}\nversion=1\nnRows={len(time)}\nnColumns={len(names) + 1}\ninDegrees=yes\n\n"
                "Units are S.I. units (second, meters, Newtons, ...)\n"
                "If the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no).\n\n"
                "endheader\n")
        f.write("time\t" + "\t".join(names) + "\n")
        for t, row in zip(time, data):
            f.write(f"{t:.4f}\t" + "\t".join(f"{x:.6f}" for x in row) + "\n")


def write_trc(path: Path, names: list[str], time: np.ndarray, xyz: np.ndarray, name: str) -> None:
    n = len(time)
    with open(path, "w") as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{name}\nDataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n"
                f"{FS:.0f}\t{FS:.0f}\t{n}\t{len(names)}\tm\t{FS:.0f}\t1\t{n}\n")
        f.write("Frame#\tTime\t" + "\t\t\t".join(names) + "\t\t\n")
        f.write("\t\t" + "\t".join(f"X{i + 1}\tY{i + 1}\tZ{i + 1}" for i in range(len(names))) + "\n\n")
        for k in range(n):
            f.write(f"{k + 1}\t{time[k]:.4f}\t" + "\t".join(f"{x:.5f}" for x in xyz[k].ravel()) + "\n")


def export(tr: Trial, source: str = "auto", align: bool = True, res=None) -> Path | None:
    """res: run() の返値（省略時は run を実行）。"""
    res = res or run(tr, source, align)
    if res is None:
        return None
    df, ctx, _ = res
    psi, rot, pos, L, TAU, K = ctx["psi"], ctx["rot"], ctx["pos_G"], ctx["L"], ctx["tau"], ctx["kick"]
    f_ = np.array([np.cos(psi), np.sin(psi), 0.0])
    l_ = np.array([-np.sin(psi), np.cos(psi), 0.0])
    C = np.array([f_, Z, -l_])  # G -> OpenSim(x正面, y上, z右)
    cvt = lambda r: R.from_matrix(C @ r.as_matrix() @ C.T)
    seg = {"pelvis": "pelvis", "femur_r": "up_R", "tibia_r": "leg_R", "foot_r": "foot_R",
           "femur_l": "up_L", "tibia_l": "leg_L", "foot_l": "foot_L"}
    tgt = {k: list(cvt(rot[v])) for k, v in seg.items()}  # 各時刻のR配列
    n = len(TAU)

    hip_w = float(np.linalg.norm(pos["hipL"] - pos["hipR"], axis=1).mean())
    m = make_scaled_model(L, hip_w)
    axes = {c: effective_axis(m, c, b) for c, b in
            [("pelvis_tilt", "pelvis"), ("pelvis_list", "pelvis"), ("pelvis_rotation", "pelvis"),
             *[(f"hip_{a}_{s}", f"femur_{s}") for s in "rl" for a in ("flexion", "adduction", "rotation")],
             *[(f"knee_angle_{s}", f"tibia_{s}") for s in "rl"], *[(f"ankle_angle_{s}", f"talus_{s}") for s in "rl"]]}

    st0 = m.initSystem()
    m.realizePosition(st0)
    unit = lambda v: v / np.linalg.norm(v)
    v0 = {f"tibia_{s}": unit(body_p(m, st0, f"talus_{s}") - body_p(m, st0, f"tibia_{s}")) for s in "rl"}
    v0.update({f"foot_{s}": unit(body_p(m, st0, f"toes_{s}") - body_p(m, st0, f"talus_{s}")) for s in "rl"})

    def solve(reverse: bool) -> dict[str, np.ndarray]:
        q = {}
        ax3 = lambda names: [axes[x] for x in names]
        pel = tgt["pelvis"]
        q["pelvis"] = decompose(pel, ax3(["pelvis_tilt", "pelvis_list", "pelvis_rotation"]), reverse)
        for s in "rl":
            rel_f = [p.inv() * c for p, c in zip(pel, tgt[f"femur_{s}"])]
            q[f"hip_{s}"] = decompose(rel_f, ax3([f"hip_flexion_{s}", f"hip_adduction_{s}", f"hip_rotation_{s}"]), reverse)
            q[f"knee_{s}"] = hinge_dir(tgt[f"femur_{s}"], tgt[f"tibia_{s}"], axes[f"knee_angle_{s}"], v0[f"tibia_{s}"])
            q[f"ankle_{s}"] = hinge_dir(tgt[f"tibia_{s}"], tgt[f"foot_{s}"], axes[f"ankle_angle_{s}"], v0[f"foot_{s}"])
        return q

    names = [m.getCoordinateSet().get(i).getName() for i in range(m.getCoordinateSet().getSize())]

    def to_table(q: dict[str, np.ndarray], trans: np.ndarray) -> np.ndarray:
        tab = np.zeros((n, len(names)))
        put = lambda nm, v: tab.__setitem__((slice(None), names.index(nm)), v)
        for i, nm in enumerate(("pelvis_tilt", "pelvis_list", "pelvis_rotation")):
            put(nm, q["pelvis"][:, i])
        for k, nm in enumerate(("pelvis_tx", "pelvis_ty", "pelvis_tz")):
            put(nm, trans[:, k])
        for s in "rl":
            for i, a in enumerate(("flexion", "adduction", "rotation")):
                put(f"hip_{a}_{s}", q[f"hip_{s}"][:, i])
            put(f"knee_angle_{s}", q[f"knee_{s}"]); put(f"ankle_angle_{s}", q[f"ankle_{s}"])
        return tab

    def forward(tab: np.ndarray, rows: range) -> dict[str, np.ndarray]:
        st = m.initSystem()
        out = {b: [] for b in ("pelvis", "femur_l", "femur_r", "tibia_l", "talus_l", "toes_l", "tibia_r", "talus_r", "toes_r")}
        out_R = {b: [] for b in ("pelvis", "femur_l", "femur_r")}
        for k in rows:
            for j, nm in enumerate(names):
                m.getCoordinateSet().get(nm).setValue(st, float(tab[k, j]), False)
            m.realizePosition(st)
            for b in out:
                out[b].append(body_p(m, st, b))
            for b in out_R:
                out_R[b].append(body_R(m, st, b))
        return {**{k: np.array(v) for k, v in out.items()}, **{f"R_{k}": v for k, v in out_R.items()}}

    # 合成順序（intrinsic / extrinsic）をOpenSimのFKとの姿勢一致で選ぶ
    best = None
    sub = range(0, n, 10)
    for reverse in (False, True):
        q = solve(reverse)
        fk = forward(to_table(q, np.zeros((n, 3))), sub)
        err = max(np.degrees((fk[f"R_{b}"][i].inv() * tgt[b][k]).magnitude()) for b in ("pelvis", "femur_l", "femur_r") for i, k in enumerate(sub))
        print(f"reverse={reverse}: 骨盤・大腿の姿勢誤差(最大) {err:.2f} deg")
        if best is None or err < best[0]:
            best = (err, reverse, q)
    q = best[2]

    # 骨盤並進: キック脚の股関節中心(femur原点)を2Dの股位置に合わせる
    fk0 = forward(to_table(q, np.zeros((n, 3))), range(n))
    hipK = (C @ pos[f"hip{K}"].T).T
    trans = hipK - fk0[f"femur_{K.lower()}"]
    tab = to_table(q, trans)
    fk = forward(tab, range(n))
    P = {k: (C @ v.T).T for k, v in pos.items()}
    print("OpenSim FK vs IMU-FK 位置差RMS[mm]: " + ", ".join(
        f"{lbl} {np.sqrt(np.mean(np.sum((fk[b] - P[k]) ** 2, 1))) * 1000:.0f}"
        for lbl, b, k in (("股R", "femur_r", "hipR"), ("膝L", "tibia_l", "kneeL"), ("足首L", "talus_l", "ankleL"),
                          ("つま先L", "toes_l", "toeL"), ("膝R", "tibia_r", "kneeR"), ("足首R", "talus_r", "ankleR"),
                          ("つま先R", "toes_r", "toeR"))))

    out = ctx["out_dir"] / "opensim"
    out.mkdir(parents=True, exist_ok=True)
    if not copy_geometry(out / "Geometry"):
        print("警告: Geometry(メッシュ)を用意できませんでした。config/paths.local.toml に geometry_dir を書くか、"
              ".osimと同じフォルダに Geometry/ を置いてください(OpenSim本体の Geometry フォルダでも可)")
    stem = f"{tr.name}_imu_hscamera"
    m.setName(stem)
    m.printToXML(str(out / f"{stem}.osim"))
    deg = np.array([1.0 if nm in ("pelvis_tx", "pelvis_ty", "pelvis_tz") else 180 / np.pi for nm in names])
    time = TAU + IMPACT_T
    write_mot(out / f"{stem}.mot", names, time, tab * deg, stem)
    mk = [("LHip", "hipL"), ("LKnee", "kneeL"), ("LAnkle", "ankleL"), ("LBigToe", "toeL"),
          ("RHip", "hipR"), ("RKnee", "kneeR"), ("RAnkle", "ankleR"), ("RBigToe", "toeR")]
    write_trc(out / f"{stem}.trc", [a for a, _ in mk], time, np.stack([P[b] for _, b in mk], 1), f"{stem}.trc")
    print(f"saved: {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", default="kurogi_t1")
    ap.add_argument("--source", default="auto", choices=["auto", "pose2d", "kinovea"])
    ap.add_argument("--no-align", action="store_true")
    a = ap.parse_args()
    export(load_trials()[a.trial], a.source, not a.no_align)
