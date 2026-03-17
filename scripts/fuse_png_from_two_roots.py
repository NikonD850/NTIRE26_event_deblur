#!/usr/bin/env python3

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


DEFAULT_ROOT1 = "results/TFM4222_27DF_28SCF_18CMF_0204_4patch_test_net_g_625000_0.5_500000_0.5"
DEFAULT_ROOT2 = "results/TFM4222_27DF_28SCF_18CMF_0204_4patch_test_net_g_625000_0.8_500000_0.2"
DEFAULT_SUBDIR = "visualization/highrev-test"
DEFAULT_NUM_WORKERS = min(64, max(1, (os.cpu_count() or 1) * 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blend PNG images from two result roots with weighted averaging."
    )
    parser.add_argument("--root1", default=DEFAULT_ROOT1, help="First result root directory.")
    parser.add_argument("--root2", default=DEFAULT_ROOT2, help="Second result root directory.")
    parser.add_argument(
        "--subdir",
        default=DEFAULT_SUBDIR,
        help="Relative image folder under each root, e.g. visualization/highrev-test.",
    )
    parser.add_argument("--w1", type=float, default=0.8, help="Weight for root1.")
    parser.add_argument("--w2", type=float, default=0.2, help="Weight for root2.")
    parser.add_argument(
        "--output-root",
        default="",
        help="Output root directory. If empty, create a folder in results/ automatically.",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        default=9,
        help="PNG compression level, valid range: 0-9. 9 is max compression.",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow writing into existing output directory.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of worker threads used for PNG read/blend/write.",
    )
    return parser.parse_args()


def resolve_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else repo_root / path


def format_weight(weight: float) -> str:
    return "{:.4f}".format(weight).rstrip("0").rstrip(".")


def collect_pngs(image_root: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for path in image_root.rglob("*.png"):
        rel = path.relative_to(image_root).as_posix()
        mapping[rel] = path
    return mapping


def blend_uint_or_float(img1: np.ndarray, img2: np.ndarray, w1: float, w2: float) -> np.ndarray:
    mixed = img1.astype(np.float32) * w1 + img2.astype(np.float32) * w2
    if np.issubdtype(img1.dtype, np.integer):
        info = np.iinfo(img1.dtype)
        mixed = np.clip(np.rint(mixed), info.min, info.max)
    return mixed.astype(img1.dtype)


def process_one_png(
    rel: str,
    pngs1: Dict[str, Path],
    pngs2: Dict[str, Path],
    output_img_root: Path,
    w1: float,
    w2: float,
    png_compression: int,
) -> Tuple[bool, str]:
    img1 = cv2.imread(str(pngs1[rel]), cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(str(pngs2[rel]), cv2.IMREAD_UNCHANGED)

    if img1 is None or img2 is None:
        return False, "read_failed"
    if img1.shape != img2.shape or img1.dtype != img2.dtype:
        return False, "shape_or_dtype_mismatch"

    mixed = blend_uint_or_float(img1, img2, w1, w2)

    out_path = output_img_root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(out_path),
        mixed,
        [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)],
    )
    if not ok:
        return False, "write_failed"
    return True, "ok"


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    if args.w1 < 0 or args.w2 < 0:
        raise ValueError("Weights must be non-negative.")
    if abs((args.w1 + args.w2) - 1.0) > 1e-6:
        raise ValueError("w1 + w2 must be exactly 1.0.")
    if args.png_compression < 0 or args.png_compression > 9:
        raise ValueError("png-compression must be in [0, 9].")
    if args.num_workers <= 0:
        raise ValueError("num-workers must be a positive integer.")

    root1 = resolve_path(repo_root, args.root1)
    root2 = resolve_path(repo_root, args.root2)
    img_dir1 = root1 / args.subdir
    img_dir2 = root2 / args.subdir

    if not img_dir1.exists():
        raise FileNotFoundError("Image directory not found: {}".format(img_dir1))
    if not img_dir2.exists():
        raise FileNotFoundError("Image directory not found: {}".format(img_dir2))

    pngs1 = collect_pngs(img_dir1)
    pngs2 = collect_pngs(img_dir2)
    common = sorted(set(pngs1.keys()) & set(pngs2.keys()))

    if not common:
        raise RuntimeError("No common PNG files found between the two roots.")

    if args.output_root:
        output_root = resolve_path(repo_root, args.output_root)
    else:
        name1 = root1.name
        name2 = root2.name
        output_name = "{}_{}_{}_{}".format(
            name1,
            format_weight(args.w1),
            name2,
            format_weight(args.w2),
        )
        output_root = repo_root / "results" / output_name

    if output_root.exists() and not args.allow_overwrite:
        raise FileExistsError(
            "Output already exists: {}. Add --allow-overwrite to continue.".format(output_root)
        )

    output_img_root = output_root / args.subdir
    output_img_root.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(
                process_one_png,
                rel,
                pngs1,
                pngs2,
                output_img_root,
                args.w1,
                args.w2,
                args.png_compression,
            )
            for rel in common
        ]

        for future in as_completed(futures):
            try:
                ok, _ = future.result()
            except Exception:
                ok = False
            if ok:
                saved += 1
            else:
                skipped += 1

    meta_path = output_root / "blend_meta.txt"
    meta_text = [
        "root1={}".format(root1),
        "root2={}".format(root2),
        "subdir={}".format(args.subdir),
        "w1={}".format(args.w1),
        "w2={}".format(args.w2),
        "common_images={}".format(len(common)),
        "saved_images={}".format(saved),
        "skipped_images={}".format(skipped),
        "png_compression={}".format(args.png_compression),
        "num_workers={}".format(args.num_workers),
    ]
    meta_path.write_text("\n".join(meta_text) + "\n", encoding="utf-8")

    print("[INFO] root1={}".format(root1))
    print("[INFO] root2={}".format(root2))
    print("[INFO] common_images={}".format(len(common)))
    print("[INFO] saved_images={}".format(saved))
    print("[INFO] skipped_images={}".format(skipped))
    print("[INFO] png_compression={}".format(args.png_compression))
    print("[INFO] num_workers={}".format(args.num_workers))
    print("[OUT ] {}".format(output_root))


if __name__ == "__main__":
    main()
