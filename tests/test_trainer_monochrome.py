import numpy as np
from PIL import Image

def test_monochrome_dataset_quantises_levels(tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    # Create a simple colour gradient image to exercise the conversion path.
    gradient = Image.new("RGB", (256, 256))
    pixels = gradient.load()
    for y in range(256):
        for x in range(256):
            pixels[x, y] = (x % 256, y % 256, (x + y) % 256)
    sample_path = source_dir / "sample.png"
    gradient.save(sample_path)

    from wqv_upscale_trainer.data import SyntheticDegradationDataset

    dataset = SyntheticDegradationDataset(
        [sample_path],
        scale=2,
        base_resolution=120,
        patches_per_image=1,
        seed=123,
        augment=False,
        monochrome_style=True,
        monochrome_levels=16,
        monochrome_noise=0.02,
    )

    sample = dataset[0]
    lr = sample["lr"].numpy()
    hr = sample["hr"].numpy()

    # Channels should be identical (monochrome replicated to RGB).
    assert np.allclose(lr[0], lr[1]) and np.allclose(lr[0], lr[2])
    assert np.allclose(hr[0], hr[1]) and np.allclose(hr[0], hr[2])

    # Values should be quantised to the configured number of levels.
    def _count_levels(tensor):
        values = np.unique(np.round(tensor * 255).astype(int))
        return len(values)

    assert _count_levels(lr[0]) <= 16
    assert _count_levels(hr[0]) <= 16
