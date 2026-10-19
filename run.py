"""どこに置いても動くランチャー: python run.py run --session S / python run.py calib ..."""
import sys
from pathlib import Path

if sys.platform == "win32":
    # pipで入れたCUDA/cuDNN(nvidia-*-cu12)のDLLをonnxruntime/torchが見つけられるようにする
    import os
    import site

    for sp in [*site.getsitepackages(), site.getusersitepackages()]:
        for d in (Path(sp) / "nvidia").glob("*/bin"):
            os.add_dll_directory(str(d))
            os.environ["PATH"] = f"{d};{os.environ['PATH']}"

try:
    import torch  # noqa: F401  Windowsでは cv2/onnxruntime より先に読み込まないとDLL初期化(WinError 1114)で失敗することがある
except Exception as e:  # torch無し/不調でもキーポイント検出自体は動く(ボール検出のみ不可)
    print(f"[警告] torch を読み込めません: {e}", flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

try:
    from pose2d.cli import main
except ModuleNotFoundError as e:
    sys.exit(f"必要なパッケージがありません: {e.name}\n  pip install -r requirements.txt")

main()
