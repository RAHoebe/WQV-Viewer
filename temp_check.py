from pathlib import Path
import sys
import tempfile

from PIL import Image
from pytest import MonkeyPatch

root = Path(__file__).resolve().parent
sys.path.append(str(root))
sys.path.append(str(root.parent))

from tests.test_upscaling import _configure_fake_backend, _make_test_variant
from wqv_viewer.upscaling import RealESRGANUpscaler

def main():
    tmp = Path(tempfile.mkdtemp())
    upscaler = RealESRGANUpscaler(_make_test_variant())
    upscaler._model_dir = tmp
    attempts = _configure_fake_backend(MonkeyPatch(), tmp, cuda_available=True, cuda_fail=True)
    try:
        upscaler.upscale(Image.new("RGB", (4, 4), color="white"), 2)
    except RuntimeError:
        pass
    print(attempts)

if __name__ == "__main__":
    main()
