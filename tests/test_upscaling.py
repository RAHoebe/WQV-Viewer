import numpy as np  # type: ignore
from PIL import Image
import pytest  # type: ignore
from typing import Optional, Set, Tuple

from wqv_viewer.upscaling import (
    ALLOWED_CONVENTIONAL_SCALES,
    PillowUpscaler,
    ai_upscalers,
    available_upscalers,
    conventional_upscalers,
    RealESRGANModelSpec,
    RealESRGANUpscaler,
    RealESRGANVariantSpec,
    RealESRGANWeightSpec,
    upscale_image,
    upscale_sequence,
)


def _make_test_variant() -> RealESRGANVariantSpec:
    return RealESRGANVariantSpec(
        id="test-variant",
        label="Test Variant",
        models={
            2: RealESRGANModelSpec(
                weights=(
                    RealESRGANWeightSpec(
                        "dummy",
                        ("https://example.com/dummy.pth",),
                    ),
                ),
                arch="rrdbnet",
                arch_args=dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
            )
        },
    )


def _configure_fake_backend(
    monkeypatch,
    tmp_path,
    *,
    cuda_available: bool,
    cuda_fail: bool,
    enhance_failures: Optional[Set[Tuple[str, bool, int]]] = None,
    enhance_invalid: Optional[Set[Tuple[str, bool, int]]] = None,
):
    def fake_ensure_weights(self, weights, loader):
        paths = []
        for weight in weights:
            path = tmp_path / f"{weight.model_name}.pth"
            path.write_bytes(b"0")
            paths.append(path)
        return paths

    attempts = []

    class FakeUpsampler:
        def __init__(self, *, tile, half, device, **kwargs):
            attempts.append((device, half, tile))
            if device == "cuda" and cuda_fail:
                raise RuntimeError("cuda failure")
            self.device = device
            self.tile = tile
            self.half = half

        def enhance(self, lr, outscale):
            h, w = lr.shape[:2]
            combo = (self.device, self.half, self.tile)
            if enhance_failures and combo in enhance_failures:
                raise RuntimeError("enhance failure")
            if enhance_invalid and combo in enhance_invalid:
                invalid = np.full((h * outscale, w * outscale, 3), np.nan, dtype=np.float32)
                return invalid, None
            return np.zeros((h * outscale, w * outscale, 3), dtype=np.float32), None

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return cuda_available

    class FakeTorch:
        cuda = FakeCuda()

    def fake_backend(self):
        backend = {
            "RealESRGANer": FakeUpsampler,
            "RRDBNet": lambda **_kwargs: object(),
            "SRVGGNetCompact": lambda **_kwargs: object(),
            "load_file_from_url": lambda *args, **kwargs: None,
            "torch": FakeTorch,
        }
        self._backend = backend
        return backend

    monkeypatch.setattr(RealESRGANUpscaler, "_ensure_weights", fake_ensure_weights, raising=False)
    monkeypatch.setattr(RealESRGANUpscaler, "_ensure_backend", fake_backend, raising=False)
    return attempts


def test_pillow_upscaler_scales_to_expected_size() -> None:
    base_image = Image.new("L", (8, 6), color=128)
    upscaler = PillowUpscaler("nearest", "Nearest", Image.NEAREST)

    for scale in ALLOWED_CONVENTIONAL_SCALES:
        result = upscaler.upscale(base_image, scale)
        assert result.size == (base_image.width * scale, base_image.height * scale)


def test_conventional_upscalers_are_pillow() -> None:
    methods = conventional_upscalers()
    assert methods, "Expected at least one conventional upscaler"
    for method in methods:
        assert isinstance(method, PillowUpscaler)


def test_ai_upscalers_provide_unique_ids() -> None:
    methods = ai_upscalers()
    assert methods, "Expected at least one AI upscaler"
    ids = [method.id for method in methods]
    assert len(ids) == len(set(ids)), "AI upscaler identifiers should be unique"


def test_available_upscalers_concatenates_all_methods() -> None:
    total = available_upscalers()
    assert len(total) == len(conventional_upscalers()) + len(ai_upscalers())


def test_real_esrgan_scales() -> None:
    method = ai_upscalers()[0]
    assert tuple(method.supported_scales()) == (2, 4)


def test_upscale_image_validates_scale() -> None:
    base_image = Image.new("RGB", (4, 4), color="white")
    method = conventional_upscalers()[0]
    invalid_scale = max(ALLOWED_CONVENTIONAL_SCALES) + 1
    with pytest.raises(ValueError):
        upscale_image(method, base_image, invalid_scale)


def test_all_conventional_and_ai_combinations() -> None:
    base_image = Image.new("RGB", (8, 6), color="white")
    conventional = conventional_upscalers()[0]
    ai_variants = ai_upscalers()

    for conventional_scale in ALLOWED_CONVENTIONAL_SCALES:
        for ai_variant in ai_variants:
            for ai_scale in ai_variant.supported_scales():
                pipeline = [
                    (conventional, conventional_scale),
                    (ai_variant, ai_scale),
                ]

                result = upscale_sequence(base_image, pipeline)

                expected_width = base_image.width * conventional_scale * ai_scale
                expected_height = base_image.height * conventional_scale * ai_scale

                assert result.size == (expected_width, expected_height)
                assert result.mode == "RGB"


def test_realesrgan_tile_and_device_fallback(monkeypatch, tmp_path) -> None:
    variant = _make_test_variant()
    upscaler = RealESRGANUpscaler(variant)
    upscaler._model_dir = tmp_path
    attempts = _configure_fake_backend(monkeypatch, tmp_path, cuda_available=True, cuda_fail=True)
    result = upscaler.upscale(Image.new("RGB", (4, 4), color="white"), 2)

    assert result.size == (8, 8)
    expected_gpu_attempts = [
        ("cuda", half, tile)
        for tile in (256, 192, 128, 96, 64, 0)
        for half in (False, True)
    ]
    assert attempts[:-1] == expected_gpu_attempts
    assert attempts[-1] == ("cpu", False, 256)
    assert len(attempts) == len(expected_gpu_attempts) + 1
    assert upscaler.describe_backend() == "CPU FP32, tile 256"


def test_realesrgan_forced_cpu_skips_cuda(monkeypatch, tmp_path) -> None:
    variant = _make_test_variant()
    upscaler = RealESRGANUpscaler(variant)
    upscaler._model_dir = tmp_path
    attempts = _configure_fake_backend(monkeypatch, tmp_path, cuda_available=True, cuda_fail=True)

    upscaler.set_device_policy("cpu")
    result = upscaler.upscale(Image.new("RGB", (4, 4), color="white"), 2)

    assert result.size == (8, 8)
    assert attempts == [("cpu", False, 256)]
    assert upscaler.device_policy() == "cpu"
    assert upscaler.describe_backend() == "CPU FP32, tile 256"


def test_realesrgan_forced_gpu_requires_cuda(monkeypatch, tmp_path) -> None:
    variant = _make_test_variant()
    upscaler = RealESRGANUpscaler(variant)
    upscaler._model_dir = tmp_path
    attempts = _configure_fake_backend(monkeypatch, tmp_path, cuda_available=False, cuda_fail=False)

    upscaler.set_device_policy("gpu")
    with pytest.raises(RuntimeError, match="GPU upscaling is not available"):
        upscaler.upscale(Image.new("RGB", (4, 4), color="white"), 2)

    assert attempts == []


def test_realesrgan_forced_gpu_does_not_fallback_to_cpu(monkeypatch, tmp_path) -> None:
    variant = _make_test_variant()
    upscaler = RealESRGANUpscaler(variant)
    upscaler._model_dir = tmp_path
    attempts = _configure_fake_backend(monkeypatch, tmp_path, cuda_available=True, cuda_fail=True)

    upscaler.set_device_policy("gpu")
    with pytest.raises(RuntimeError, match="Unable to initialise Real-ESRGAN on the GPU"):
        upscaler.upscale(Image.new("RGB", (4, 4), color="white"), 2)

    expected_gpu_attempts = [
        ("cuda", half, tile)
        for tile in (256, 192, 128, 96, 64, 0)
        for half in (False, True)
    ]
    assert attempts == expected_gpu_attempts


def test_realesrgan_enhance_failure_triggers_retry(monkeypatch, tmp_path) -> None:
    variant = _make_test_variant()
    upscaler = RealESRGANUpscaler(variant)
    upscaler._model_dir = tmp_path
    gpu_attempts: Set[Tuple[str, bool, int]] = {
        ("cuda", half, tile)
        for tile in (256, 192, 128, 96, 64, 0)
        for half in (False, True)
    }
    expected_gpu_attempts = [
        ("cuda", half, tile)
        for tile in (256, 192, 128, 96, 64, 0)
        for half in (False, True)
    ]
    attempts = _configure_fake_backend(
        monkeypatch,
        tmp_path,
        cuda_available=True,
        cuda_fail=False,
        enhance_failures=gpu_attempts,
    )

    result = upscaler.upscale(Image.new("RGB", (4, 4), color="white"), 2)

    assert result.size == (8, 8)
    assert attempts[:-1] == expected_gpu_attempts
    assert attempts[-1] == ("cpu", False, 256)
    assert upscaler.describe_backend() == "CPU FP32, tile 256"


def test_realesrgan_invalid_output_triggers_retry(monkeypatch, tmp_path) -> None:
    variant = _make_test_variant()
    upscaler = RealESRGANUpscaler(variant)
    upscaler._model_dir = tmp_path
    invalid_attempts: Set[Tuple[str, bool, int]] = {("cuda", False, 256)}
    attempts = _configure_fake_backend(
        monkeypatch,
        tmp_path,
        cuda_available=True,
        cuda_fail=False,
        enhance_invalid=invalid_attempts,
    )

    result = upscaler.upscale(Image.new("RGB", (4, 4), color="white"), 2)

    assert result.size == (8, 8)
    assert attempts[0] == ("cuda", False, 256)
    assert attempts[1] == ("cuda", True, 256)
    assert len(attempts) == 2
    assert upscaler.describe_backend() == "CUDA FP16, tile 256"
