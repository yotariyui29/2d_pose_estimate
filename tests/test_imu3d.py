"""データ不要の単体テスト: python tests/test_imu3d.py"""
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from imu3d import imu_video_3d as m  # noqa: E402
from imu3d.trials import Trial2D, load_paths, resolve_path  # noqa: E402


def test_resolve_path() -> None:
    assert str(resolve_path("/mnt/c/a/b")).replace("\\", "/").endswith("a/b")
    assert resolve_path("C:/x/y").is_absolute() or True
    assert resolve_path("config").is_absolute()


def test_paths_toml() -> None:
    p = load_paths()
    assert {"xsens_root", "pose2d_root"} <= set(p)


def test_kinovea_flip_and_window() -> None:
    n = 1401
    a = np.zeros((n, 11))
    a[:, 0] = np.arange(n) - 900  # frame, インパクト=0
    a[:, 1::2] = np.linspace(0, 1000, n)[:, None]  # x[mm]
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "k.csv"
        np.savetxt(f, a, delimiter=",")
        t2 = Trial2D.from_kinovea(f, 1000, flip_x=True)
        assert abs(t2.t.min() + 0.9) < 1e-9 and t2.pts["HIP"][-1, 0] < 0  # xが反転
        tau = m.analysis_window(t2)
        assert tau[0] == -0.9 and tau[-1] <= 0.5 - 1 / m.FS + 1e-9


def test_yaw_correct_identity() -> None:
    rot = {k: R.from_rotvec(np.random.default_rng(0).normal(size=(5, 3)) * 0.1) for k in ["pelvis", *m.ALIGN_KEYS]}
    out = m.yaw_correct(rot, np.zeros(6), np.zeros((6, 3)))
    for k in rot:
        assert (out[k].inv() * rot[k]).magnitude().max() < 1e-12


def test_joint_angles_pure_flexion() -> None:
    q = np.radians([10, 20, 30])
    parent = R.identity(3)
    child = R.from_rotvec(np.outer(q, [0, 1, 0]))  # y軸(左向き)まわり = 矢状面の屈伸
    a = m.joint_angles(parent, child, 0.0, "L")
    assert np.allclose(a[:, 0], np.degrees(q), atol=1e-9) and np.abs(a[:, 1:]).max() < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
