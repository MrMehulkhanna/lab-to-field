"""Datasets, splits and augmentations.

Three rules keep the evaluation honest:
1. PlantVillage has many photos of the *same physical leaf*. Train/val/test are split by
   leaf_id, so the model is never tested on a leaf it has already seen.
2. PlantDoc test is the field exam. It is never used for training or model selection.
3. Only the 28 classes the two datasets share are used (see config.CLASS_MAP).
"""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .config import IMAGENET_MEAN, IMAGENET_STD, PD_CACHE, PD_DIR, PD_TO_IDX, PV_DIR, PV_TO_IDX

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------- PlantVillage (lab)

def _leaf_id(rel_path: str, leaf_map: dict) -> str:
    """Same leaf-grouping rule as the official Hugging Face loader (mohanty/PlantVillage)."""
    class_name, file_name = rel_path.split("/")[-2:]
    ident = file_name.replace("_final_masked", "")
    if "___" in ident:
        ident = ident.split("___")[-1]
    ident = ident.split("copy")[0]
    for ext in (".jpg", ".JPG", ".png", ".PNG"):
        ident = ident.replace(ext, "")
    key = ident.lower().strip()
    hits = leaf_map.get(key, [])
    if len(hits) == 1:
        return hits[0]
    for h in hits:
        if class_name in h:
            return h
    return f"fallback_{class_name}_{key}"


def load_plantvillage(split: str) -> pd.DataFrame:
    """Official leaf-grouped split ('train' or 'test'), restricted to the shared classes.

    If subset.txt exists (written by scripts/download_plantvillage_subset.py), only those
    photos are used: whole leaves sampled up to ~700 train / ~150 test photos per class.
    """
    leaf_map = json.loads((PV_DIR / "leaf_grouping" / "leaf-map.json").read_text())
    subset_file = PV_DIR / "subset.txt"
    subset = set(subset_file.read_text().split("\n")) if subset_file.exists() else None
    rows = []
    for rel in (PV_DIR / "splits" / f"color_{split}.txt").read_text().split("\n"):
        rel = rel.strip()
        if not rel or (subset is not None and rel not in subset):
            continue
        pv_class = rel.split("/")[-2]
        if pv_class in PV_TO_IDX:
            rows.append({"path": str(PV_DIR / rel), "label": PV_TO_IDX[pv_class],
                         "leaf_id": _leaf_id(rel, leaf_map)})
    return pd.DataFrame(rows)


def segmented_path(color_path: str) -> Path:
    """The background-removed copy of a PlantVillage color image."""
    p = Path(color_path.replace("/color/", "/segmented/"))
    return p.with_name(p.stem + "_final_masked.jpg")


def drop_shared_leaves(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Remove training photos of any leaf that also appears in the test set.

    The official PlantVillage train/test split is grouped by leaf, but a few leaves still
    end up on both sides, so we remove them from training to be safe.
    """
    return train[~train["leaf_id"].isin(set(test["leaf_id"]))].reset_index(drop=True)


def split_by_group(df: pd.DataFrame, val_frac: float = 0.1, seed: int = 0):
    """Hold out whole leaves (groups), never individual photos."""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr, va = next(gss.split(df, groups=df["leaf_id"]))
    return df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True)


def cap_per_class(df: pd.DataFrame, n: int | None, seed: int = 0) -> pd.DataFrame:
    """Keep at most n images per class: faster experiments and less class imbalance."""
    if n is None:
        return df
    parts = [g.sample(min(len(g), n), random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts).reset_index(drop=True)


# ---------------------------------------------------------------- PlantDoc (field)

def cache_plantdoc(size: int = 256) -> None:
    """Resize PlantDoc once (shorter side = size). Originals are often several megapixels."""
    for split in ("train", "test"):
        for cls_dir in sorted((PD_DIR / split).iterdir()):
            out_dir = PD_CACHE / split / cls_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in cls_dir.iterdir():
                out = out_dir / (f.stem + ".jpg")
                if f.suffix.lower() not in IMG_EXTS or out.exists():
                    continue
                try:
                    im = Image.open(f).convert("RGB")
                except Exception:  # a few PlantDoc files are corrupt or not images
                    continue
                im.thumbnail((size * 4, size * 4))
                w, h = im.size
                s = size / min(w, h)
                im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC).save(out, quality=95)


def find_plantdoc_duplicates(max_distance: int = 4) -> list[str]:
    """PlantDoc was scraped from the web, so the same photo can be in train and test.

    Compare perceptual hashes (pHash: a 64-bit fingerprint that survives resizing and
    re-compression). Train photos within `max_distance` bits of any test photo are returned,
    and load_plantdoc("train") drops them. The result is cached in data/plantdoc_duplicates.json.
    """
    import imagehash

    cache = PD_CACHE.parent / "plantdoc_duplicates.json"
    if cache.exists():
        return json.loads(cache.read_text())
    phash = lambda p: imagehash.phash(Image.open(p).convert("RGB"))
    test = [phash(p) for p in sorted((PD_CACHE / "test").rglob("*.jpg"))]
    dupes = [str(p) for p in sorted((PD_CACHE / "train").rglob("*.jpg"))
             if min(phash(p) - t for t in test) <= max_distance]
    cache.write_text(json.dumps(dupes, indent=1))
    return dupes


def load_plantdoc(split: str) -> pd.DataFrame:
    skip = set(find_plantdoc_duplicates()) if split == "train" else set()
    rows = []
    for cls_dir in sorted((PD_CACHE / split).iterdir()):
        if cls_dir.name in PD_TO_IDX:
            for f in sorted(cls_dir.glob("*.jpg")):
                if str(f) not in skip:
                    rows.append({"path": str(f), "label": PD_TO_IDX[cls_dir.name], "leaf_id": f.stem})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- augmentation

class BackgroundReplacer:
    """Paste the segmented PlantVillage leaf onto a new background.

    PlantVillage backgrounds are all similar grey/black boards, and models learn to use
    them as a shortcut. Replacing the background forces the model to look at the leaf.
      source="synthetic": random colors, gradients and blurred noise (no field data used)
      source="field"    : random crops of PlantDoc *train* photos, used without labels
    """

    def __init__(self, source: str = "synthetic", p: float = 0.8, field_paths: list[str] | None = None):
        self.source, self.p, self.field_paths = source, p, field_paths or []

    def _background(self, size) -> Image.Image:
        w, h = size
        if self.source == "field" and self.field_paths:
            bg = Image.open(random.choice(self.field_paths)).convert("RGB")
            scale = random.uniform(1.0, 2.5)
            bg = bg.resize((int(bg.width * scale), int(bg.height * scale)))
            x, y = random.randint(0, bg.width - w), random.randint(0, bg.height - h)
            return bg.crop((x, y, x + w, y + h)).filter(ImageFilter.GaussianBlur(random.uniform(0, 2)))
        kind = random.random()
        if kind < 0.33:  # flat color
            return Image.new("RGB", size, tuple(np.random.randint(0, 256, 3).tolist()))
        if kind < 0.66:  # two-color gradient
            a, b = np.random.randint(0, 256, (2, 3))
            t = np.linspace(0, 1, w)[None, :, None]
            return Image.fromarray(((1 - t) * a + t * b).repeat(h, 0).astype(np.uint8))
        noise = np.random.randint(0, 256, (h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)  # blotchy texture
        return Image.fromarray(noise).resize(size, Image.BICUBIC).filter(ImageFilter.GaussianBlur(3))

    def __call__(self, color_path: str) -> Image.Image:
        img = Image.open(color_path).convert("RGB")
        seg = segmented_path(color_path)
        if random.random() > self.p or not seg.exists():
            return img
        leaf = Image.open(seg).convert("RGB")
        # Segmented images have a black background: anything brighter than ~black is leaf.
        mask = Image.fromarray(((np.asarray(leaf).max(2) > 20) * 255).astype(np.uint8))
        mask = mask.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1))
        out = self._background(leaf.size)
        out.paste(leaf, (0, 0), mask)
        return out


def build_transforms(kind: str, img_size: int = 224, train: bool = True):
    """Named augmentation recipes compared in notebook 04.

    none     : resize + center crop only
    basic    : random crop/scale + horizontal and vertical flips
    color    : basic + lighting, blur and rotation (imitates phone photos outdoors)
    randaug  : basic + RandAugment (random strong operations)
    bg_*     : background replacement (done in the Dataset) + color
    """
    normalize = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train or kind == "none":
        return T.Compose([T.Resize(int(img_size * 1.14)), T.CenterCrop(img_size), *normalize])
    basic = [T.RandomResizedCrop(img_size, scale=(0.5, 1.0)), T.RandomHorizontalFlip(), T.RandomVerticalFlip()]
    color = [T.ColorJitter(0.4, 0.4, 0.4, 0.05), T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=0.3),
             T.RandomRotation(20)]
    if kind == "basic":
        return T.Compose([*basic, *normalize])
    if kind in ("color", "bg_synthetic", "bg_field"):
        return T.Compose([*basic, *color, *normalize])
    if kind == "randaug":
        return T.Compose([*basic, T.RandAugment(num_ops=2, magnitude=9), *normalize])
    raise ValueError(kind)


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform, bg: BackgroundReplacer | None = None):
        self.paths, self.labels = df["path"].tolist(), df["label"].tolist()
        self.transform, self.bg = transform, bg

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = self.bg(self.paths[i]) if self.bg else Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def make_loader(df, transform, batch_size=64, shuffle=False, bg=None, workers=8):
    return DataLoader(ImageDataset(df, transform, bg), batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
                      drop_last=shuffle)


def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
