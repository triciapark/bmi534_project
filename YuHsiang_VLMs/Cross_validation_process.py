#!/usr/bin/env python3
"""
Generate cross-validation folds for MedXpertQA.

Fold 0 reuses the current split under processed/train_splits as the train set,
and builds its test set from the remaining examples found in
MedXpertQA_by_dimension/body_system. Additional folds are created with
deterministic (ordered, non-shuffled) k-fold splits per body system. Train
splits are saved as Hugging Face datasets, while test splits are stored as JSON.
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from datasets import Dataset, DatasetDict, Features, Image, Value
from PIL import Image as PILImage
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MedXpertQA cross-validation folds.")
    parser.add_argument(
        "--body-system-dir",
        type=Path,
        default=Path("MedXpertQA_by_dimension/body_system"),
        help="Directory containing the full body-system JSON files.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("MedXpertQA_images"),
        help="Directory that stores the raw image files.",
    )
    parser.add_argument(
        "--existing-train-dir",
        type=Path,
        default=Path("processed/train_splits"),
        help="Current train JSON split; reused as fold 0 train set.",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=5,
        help="Total number of folds to generate (fold 0 is reserved for the existing split).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed/cv_folds"),
        help="Base directory where folds will be written.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=384,
        help="Max image size (longest edge) for train dataset saving.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output directory.",
    )
    return parser.parse_args()


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_json_items(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError(f"{path} should contain a list.")
    return data


def load_per_system_json(json_dir: Path, allowed: set[str] | None = None) -> dict[str, list[dict]]:
    per_system: dict[str, list[dict]] = {}
    for json_path in sorted(json_dir.glob("*.json")):
        system = json_path.stem
        if system == "train_all":
            continue
        if allowed is not None and system not in allowed:
            continue
        per_system[system] = load_json_items(json_path)
    return per_system


def load_full_dataset(body_system_dir: Path) -> dict[str, list[dict]]:
    if not body_system_dir.is_dir():
        raise FileNotFoundError(f"Body-system directory not found: {body_system_dir}")
    per_system = load_per_system_json(body_system_dir)
    if not per_system:
        raise RuntimeError(f"No body-system JSON files found in {body_system_dir}")
    return per_system


def item_key(item: dict) -> tuple[str, str, str]:
    return item.get("image", ""), item.get("problem", ""), item.get("solution", "")


def complement(full_items: Iterable[dict], subset_items: Iterable[dict]) -> list[dict]:
    subset_keys = {item_key(item) for item in subset_items}
    return [item for item in full_items if item_key(item) not in subset_keys]


def split_items_into_folds(items: list[dict], num_folds: int) -> list[list[dict]]:
    folds: list[list[dict]] = []
    base, extra = divmod(len(items), num_folds)
    start = 0
    for i in range(num_folds):
        end = start + base + (1 if i < extra else 0)
        folds.append(items[start:end])
        start = end
    return folds


def prepare_fold_splits(
    fold_idx: int,
    num_folds: int,
    full_data: dict[str, list[dict]],
    existing_train_dir: Path,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    if fold_idx == 0:
        train_data = load_per_system_json(existing_train_dir, allowed=set(full_data.keys()))
        if not train_data:
            raise RuntimeError(f"No JSON files found in {existing_train_dir} for fold 0.")
        test_data = {sys: complement(full_data[sys], train_data.get(sys, [])) for sys in full_data}
        return train_data, test_data

    train_data: dict[str, list[dict]] = {}
    test_data: dict[str, list[dict]] = {}
    for system, items in full_data.items():
        folds = split_items_into_folds(items, num_folds)
        test_slice = folds[fold_idx]
        train_slice: list[dict] = []
        for i, fold_items in enumerate(folds):
            if i == fold_idx:
                continue
            train_slice.extend(fold_items)
        train_data[system] = train_slice
        test_data[system] = test_slice
    return train_data, test_data


def resolve_image_path(image_value: str, images_dir: Path) -> Path:
    image_name = Path(image_value).name
    resolved = images_dir / image_name
    if not resolved.exists():
        raise FileNotFoundError(f"Cannot find image file: {resolved}")
    return resolved.resolve()


def load_image_record(image_value: str, images_dir: Path, max_size: int) -> PILImage.Image:
    resolved_path = resolve_image_path(image_value, images_dir)
    img = PILImage.open(resolved_path).convert("RGB")
    if img.width > max_size or img.height > max_size:
        img.thumbnail((max_size, max_size))
    return img


def build_hf_dataset(train_items: list[dict], images_dir: Path, max_size: int) -> DatasetDict:
    records: list[dict] = []

    def load_one(item: dict) -> tuple[dict, PILImage.Image | None]:
        try:
            return item, load_image_record(item["image"], images_dir, max_size)
        except Exception as exc:
            print(f"⚠️ 图片加载失败: {item.get('image')}，错误: {exc}")
            return item, None

    with ThreadPoolExecutor() as executor:
        future_to_item = {executor.submit(load_one, item): item for item in train_items}
        for future in as_completed(future_to_item):
            item, img = future.result()
            if img is None:
                continue
            records.append(
                {
                    "image": img,
                    "problem": item["problem"],
                    "solution": item["solution"],
                }
            )

    image_values, problems, solutions = [], [], []
    for record in tqdm(records, desc="加载训练图片", unit="条"):
        image_values.append(record["image"])
        problems.append(record["problem"])
        solutions.append(record["solution"])

    features = Features({"image": Image(), "problem": Value("string"), "solution": Value("string")})
    train_dataset = Dataset.from_dict(
        {"image": image_values, "problem": problems, "solution": solutions},
        features=features,
    )
    return DatasetDict({"train": train_dataset})


def write_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)


def save_json_splits(base_dir: Path, per_system_data: dict[str, list[dict]], combined_name: str) -> list[dict]:
    combined: list[dict] = []
    for system, items in per_system_data.items():
        write_json(base_dir / f"{system}.json", items)
        combined.extend(items)
    write_json(base_dir.parent / combined_name, combined)
    return combined


def save_fold(
    fold_idx: int,
    output_dir: Path,
    train_data: dict[str, list[dict]],
    test_data: dict[str, list[dict]],
    images_dir: Path,
    max_size: int,
) -> None:
    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    json_dir = fold_dir / "json"
    train_json_dir = json_dir / "train"
    test_json_dir = json_dir / "test"

    train_items = save_json_splits(train_json_dir, train_data, "train_all.json")
    test_items = save_json_splits(test_json_dir, test_data, "test_all.json")

    dataset_dict = build_hf_dataset(train_items, images_dir, max_size)
    dataset_path = fold_dir / "train_dataset"
    dataset_dict.save_to_disk(str(dataset_path))

    stats = {
        "fold": fold_idx,
        "train_count": len(train_items),
        "test_count": len(test_items),
        "per_system_counts": {
            system: {"train": len(train_data.get(system, [])), "test": len(test_data.get(system, []))}
            for system in sorted({*train_data.keys(), *test_data.keys()})
        },
    }
    write_json(fold_dir / "stats.json", stats)
    print(f"Fold {fold_idx}: train {stats['train_count']}, test {stats['test_count']} -> {fold_dir}")


def main() -> None:
    args = parse_args()
    if args.num_folds < 2:
        raise ValueError("num_folds must be at least 2.")
    if not args.images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")

    full_data = load_full_dataset(args.body_system_dir)
    ensure_output_dir(args.output_dir, args.overwrite)

    for fold_idx in range(args.num_folds):
        train_data, test_data = prepare_fold_splits(
            fold_idx,
            args.num_folds,
            full_data,
            args.existing_train_dir,
        )
        save_fold(fold_idx, args.output_dir, train_data, test_data, args.images_dir, args.max_size)


if __name__ == "__main__":
    main()
