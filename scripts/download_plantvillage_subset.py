"""Download only the PlantVillage images this project uses.

The official zip (mohanty/PlantVillage on Hugging Face) is 2.2 GB and holds color, grayscale
and segmented copies of all 54,306 photos. We read the zip's index over HTTP and fetch just:
  - color images for up to TRAIN_PER_CLASS train and TEST_PER_CLASS test photos per shared class
  - the segmented copy of each selected train photo (needed for background replacement)
Photos are selected by whole leaf (leaf_id), so the leakage-safe split is preserved.
The chosen file list is saved to data/raw/plantvillage/subset.txt.

Usage: uv run python scripts/download_plantvillage_subset.py   (or plain python3: stdlib only)
"""
import io
import json
import random
import struct
import sys
import urllib.request
import zipfile
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PV = ROOT / "data" / "raw" / "plantvillage"
ZIP_URL = "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/data.zip"
TRAIN_PER_CLASS, TEST_PER_CLASS, SEED = 700, 150, 0


def shared_classes() -> set[str]:
    from lab_to_field.config import CLASS_MAP
    return {pv for pv, _ in CLASS_MAP.values()}


def leaf_id(rel: str, leaf_map: dict) -> str:
    class_name, file_name = rel.split("/")[-2:]
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


def select(split: str, per_class: int, classes: set[str], leaf_map: dict) -> list[str]:
    groups = defaultdict(lambda: defaultdict(list))
    for rel in (PV / "splits" / f"color_{split}.txt").read_text().split("\n"):
        rel = rel.strip()
        if rel and rel.split("/")[-2] in classes:
            groups[rel.split("/")[-2]][leaf_id(rel, leaf_map)].append(rel)
    rng, chosen = random.Random(SEED), []
    for cls in sorted(groups):
        leaves = sorted(groups[cls])
        rng.shuffle(leaves)
        n = 0
        for lf in leaves:
            if n >= per_class:
                break
            chosen += groups[cls][lf]
            n += len(groups[cls][lf])
    return chosen


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests, with a block cache (for zipfile)."""

    def __init__(self, url: str, block: int = 1 << 20):
        for attempt in range(6):
            try:
                head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
                break
            except Exception:
                if attempt == 5:
                    raise
        self.url, self.size = head.url, int(head.headers["Content-Length"])
        self.pos, self.block, self.cache = 0, block, {}

    def fetch(self, start: int, end: int) -> bytes:  # inclusive end
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        for attempt in range(6):
            try:
                return urllib.request.urlopen(req, timeout=120).read()
            except Exception:
                if attempt == 5:
                    raise

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            b = self.pos // self.block
            if b not in self.cache:
                self.cache[b] = self.fetch(b * self.block, min(self.size, (b + 1) * self.block) - 1)
            chunk = self.cache[b][self.pos - b * self.block:][:n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)


def main():
    leaf_map = json.loads((PV / "leaf_grouping" / "leaf-map.json").read_text())
    classes = shared_classes()
    train, test = select("train", TRAIN_PER_CLASS, classes, leaf_map), select("test", TEST_PER_CLASS, classes, leaf_map)
    (PV / "subset.txt").write_text("\n".join(train + test) + "\n")
    print(f"selected {len(train)} train + {len(test)} test color photos")

    remote = HTTPRangeFile(ZIP_URL)
    zf = zipfile.ZipFile(remote)
    by_name = {i.filename: i for i in zf.infolist()}
    prefix = next(n[: n.index("raw/")] for n in by_name if "raw/color/" in n)

    def seg_name(rel):
        p = rel.replace("raw/color/", "raw/segmented/")
        return p.rsplit(".", 1)[0] + "_final_masked.jpg"

    wanted = [prefix + r for r in train + test] + [prefix + seg_name(r) for r in train]
    missing = [w for w in wanted if w not in by_name]
    infos = [by_name[w] for w in wanted if w in by_name]
    print(f"{len(infos)} files to fetch ({sum(i.compress_size for i in infos) / 1e6:.0f} MB), {len(missing)} not in zip")

    def get(info):
        out = PV / info.filename[len(prefix):]
        if out.exists() and out.stat().st_size == info.file_size:
            return 0
        # One request: local header (30 bytes + name + extra field, extra usually small) + data
        span = 30 + len(info.orig_filename.encode()) + 1024 + info.compress_size
        blob = remote.fetch(info.header_offset, info.header_offset + span)
        name_len, extra_len = struct.unpack("<HH", blob[26:30])
        start = 30 + name_len + extra_len
        data = blob[start:start + info.compress_size]
        if info.compress_type == zipfile.ZIP_DEFLATED:
            data = zlib.decompress(data, -15)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        return 1

    with ThreadPoolExecutor(16) as pool:
        for k, _ in enumerate(pool.map(get, infos), 1):
            if k % 2000 == 0:
                print(f"  {k}/{len(infos)}", flush=True)
    print("done:", PV)


if __name__ == "__main__":
    main()
