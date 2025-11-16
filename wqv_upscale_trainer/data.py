"""Dataset and synthetic degradation utilities."""

from __future__ import annotations

import io
import math
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2  # type: ignore
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset

from .config import ScaleFactor

ImagePath = Path


def discover_images(source_dir: Path, patterns: Sequence[str] | None = None) -> List[ImagePath]:
    source_dir = source_dir.expanduser().resolve()
    if patterns is None:
        patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    paths: List[ImagePath] = []
    for pattern in patterns:
        paths.extend(source_dir.rglob(pattern))
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise FileNotFoundError(f"No images found in {source_dir}")
    return unique


def _load_image(path: ImagePath) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def _random_crop(img: Image.Image, size: int, rng: random.Random) -> Image.Image:
    width, height = img.size
    if width < size or height < size:
        scale = size / min(width, height)
        new_width = int(math.ceil(width * scale))
        new_height = int(math.ceil(height * scale))
        img = img.resize((new_width, new_height), Image.LANCZOS)
        width, height = img.size
    if width == size and height == size:
        return img
    left = rng.randint(0, width - size)
    top = rng.randint(0, height - size)
    return img.crop((left, top, left + size, top + size))


def _apply_blur(image: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.5:
        return image
    kernel_size = rng.choice([1, 3, 5, 7])
    if kernel_size <= 1:
        return image
    sigma = rng.uniform(0.2, 1.2)
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def _apply_noise(image: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.4:
        return image
    noise_sigma = rng.uniform(0.002, 0.01)
    noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0.0, noise_sigma, size=image.shape).astype(np.float32)
    return np.clip(image + noise, 0.0, 1.0)


def _apply_jpeg(image: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.4:
        return image
    quality = rng.randint(30, 85)
    pil_image = Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8), mode="RGB")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality, subsampling=0)
    buffer.seek(0)
    degraded = Image.open(buffer)
    return np.asarray(degraded, dtype=np.uint8).astype(np.float32) / 255.0


def _apply_color_variation(image: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.3:
        return image
    brightness = rng.uniform(0.85, 1.15)
    contrast = rng.uniform(0.85, 1.15)
    saturation = rng.uniform(0.8, 1.2)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(saturation)


def _apply_monochrome_style(
    image: Image.Image,
    rng: random.Random,
    *,
    levels: int,
    noise_strength: float,
) -> Image.Image:
    grayscale = image.convert("L")
    array = np.asarray(grayscale, dtype=np.float32) / 255.0

    if rng.random() < 0.9:
        contrast = rng.uniform(0.9, 1.2)
        array = np.clip((array - 0.5) * contrast + 0.5, 0.0, 1.0)
    if rng.random() < 0.9:
        brightness = rng.uniform(0.9, 1.1)
        array = np.clip(array * brightness, 0.0, 1.0)

    if noise_strength > 0.0:
        sigma = rng.uniform(noise_strength * 0.5, noise_strength * 1.5)
        noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0.0, sigma, size=array.shape).astype(np.float32)
        array = np.clip(array + noise, 0.0, 1.0)

    quantised = _quantize_levels(array, levels)

    if noise_strength > 0.0:
        speckle_sigma = noise_strength * 0.3
        if speckle_sigma > 0.0:
            speckle = np.random.default_rng(rng.randrange(1 << 30)).normal(0.0, speckle_sigma, size=array.shape).astype(np.float32)
            quantised = _quantize_levels(np.clip(quantised + speckle, 0.0, 1.0), levels)

    uint8 = (quantised * 255).astype(np.uint8)
    rgb = np.stack([uint8] * 3, axis=2)
    return Image.fromarray(rgb, mode="RGB")


def _quantize_levels(array: np.ndarray, levels: int) -> np.ndarray:
    levels = max(2, int(levels))
    quantised = np.round(array * (levels - 1)) / (levels - 1)
    return np.clip(quantised, 0.0, 1.0)


def synthetic_degradation(
    hr_patch: Image.Image,
    lr_size: int,
    rng: random.Random,
    *,
    monochrome_style: bool = False,
    monochrome_levels: int = 16,
    monochrome_noise: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    if monochrome_style:
        hr_patch = _apply_monochrome_style(
            hr_patch,
            rng,
            levels=monochrome_levels,
            noise_strength=monochrome_noise,
        )
    else:
        hr_patch = _apply_color_variation(hr_patch, rng)

    hr_array = np.asarray(hr_patch, dtype=np.uint8).astype(np.float32) / 255.0
    degraded = _apply_blur(hr_array, rng)
    degraded = _apply_noise(degraded, rng)
    degraded = _apply_jpeg(degraded, rng)
    degraded_pil = Image.fromarray((np.clip(degraded, 0.0, 1.0) * 255).astype(np.uint8), mode="RGB")
    interpolation = rng.choice([Image.BICUBIC, Image.BILINEAR, Image.LANCZOS, Image.BOX])
    lr_image = degraded_pil.resize((lr_size, lr_size), interpolation)
    lr_array = np.asarray(lr_image, dtype=np.uint8).astype(np.float32) / 255.0
    if monochrome_style:
        hr_array = _quantize_levels(hr_array, monochrome_levels)
        lr_array = _quantize_levels(lr_array, monochrome_levels)
    return lr_array, hr_array


class SyntheticDegradationDataset(Dataset):
    """Generates synthetic WQV-style LR/HR pairs on the fly."""

    def __init__(
        self,
        image_paths: Sequence[ImagePath],
        *,
        scale: ScaleFactor,
        base_resolution: int,
        patches_per_image: int,
        seed: int,
        augment: bool = True,
        monochrome_style: bool = False,
        monochrome_levels: int = 16,
        monochrome_noise: float = 0.02,
    ) -> None:
        self.paths = list(image_paths)
        self.scale = scale
        self.base_resolution = base_resolution
        self.target_size = base_resolution * scale
        self.patches_per_image = patches_per_image
        self.seed = seed
        self.augment = augment
        self.monochrome_style = monochrome_style
        self.monochrome_levels = max(2, int(monochrome_levels))
        self.monochrome_noise = max(0.0, float(monochrome_noise))
        self.length = len(self.paths) * patches_per_image

    def __len__(self) -> int:
        return self.length

    def _rng_for_index(self, index: int) -> random.Random:
        return random.Random(self.seed + index * 17)

    @staticmethod
    def _to_tensor(image: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(image.transpose(2, 0, 1))
        return tensor.float()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path = self.paths[index % len(self.paths)]
        rng = self._rng_for_index(index)
        image = _load_image(path)
        patch = _random_crop(image, self.target_size, rng)
        if self.augment:
            if rng.random() < 0.5:
                patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
            if rng.random() < 0.5:
                patch = patch.transpose(Image.FLIP_TOP_BOTTOM)
            rotations = rng.randint(0, 3)
            if rotations:
                patch = patch.rotate(90 * rotations)
        lr_array, hr_array = synthetic_degradation(
            patch,
            self.base_resolution,
            rng,
            monochrome_style=self.monochrome_style,
            monochrome_levels=self.monochrome_levels,
            monochrome_noise=self.monochrome_noise,
        )
        lr_tensor = self._to_tensor(lr_array)
        hr_tensor = self._to_tensor(hr_array)
        return {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "path": str(path),
        }


def split_datasets(
    image_paths: Sequence[ImagePath],
    *,
    train_split: float,
    val_split: float,
) -> Tuple[List[ImagePath], List[ImagePath], List[ImagePath]]:
    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split must be between 0 and 1")
    if not 0.0 <= val_split < 1.0:
        raise ValueError("val_split must be between 0 and 1")
    if train_split + val_split >= 1.0:
        raise ValueError("train_split + val_split must be < 1")
    total = len(image_paths)
    train_end = int(total * train_split)
    val_end = train_end + int(total * val_split)
    shuffled = list(image_paths)
    random.Random(12345).shuffle(shuffled)
    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]
    return train, val, test
