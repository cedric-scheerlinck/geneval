import argh
from tqdm import tqdm
import jsonlines
import json
import shutil
from pathlib import Path
from typing import Literal, get_args
from PIL import Image

GridSize = Literal["1x1", "2x2"]


def unenhance_prompt(prompt: str) -> str:
    """
    enhanced prompt can have blah: prefix or prompt\nblah.
    """
    if ":" in prompt:
        prompt_list = prompt.split(":")
        assert len(prompt_list) == 2
        prompt = prompt_list[1].strip()
    if "\n" in prompt:
        prompt_list = prompt.split("\n")
        assert len(prompt_list) == 2
        prompt = prompt_list[0].strip()
    return prompt


def find_matching_metadata(
    prompt_text: str, metadata_from_prompt: dict[str, tuple[int, dict]]
) -> tuple[int, dict] | None:
    prompt_text = unenhance_prompt(prompt_text)
    result = None
    for prompt_key, (line_index, metadata) in metadata_from_prompt.items():
        if prompt_key == prompt_text:
            if result is not None:
                raise ValueError(f"Multiple matching metadata found for {prompt_text}")
            result = (line_index, metadata)
    return result


def process_samples(
    src_path: Path,
    dst_root: Path,
    metadata_from_prompt: dict[str, tuple[int, dict]],
    grid: GridSize,
) -> None:
    dst_root.mkdir(parents=True, exist_ok=True)

    matched_count = 0
    unmatched_count = 0
    missing_files_count = 0

    src_sample_dirs = sorted(
        d for d in src_path.iterdir() if d.is_dir() and d.name.startswith("sample_")
    )
    for src_sample_dir in tqdm(src_sample_dirs, desc="Processing samples"):
        prompt_file = src_sample_dir / "prompt.txt"
        src_image_file = src_sample_dir / "image.png"
        try:
            with open(prompt_file, "r") as fp:
                prompt_text = fp.read().strip()
        except Exception as e:
            print(f"Warning: Failed to read prompt.txt from {src_sample_dir.name}: {e}")
            missing_files_count += 1
            continue

        match = find_matching_metadata(prompt_text, metadata_from_prompt)

        if match is None:
            print(f"Warning: No matching metadata found for {src_sample_dir.name}")
            unmatched_count += 1
            continue

        line_index, metadata = match

        dst_dir = dst_root / f"{line_index:05d}"
        dst_dir.mkdir(parents=True, exist_ok=True)

        dst_samples_dir = dst_dir / "samples"
        dst_samples_dir.mkdir(parents=True, exist_ok=True)

        dst_metadata_file = dst_dir / "metadata.jsonl"
        with jsonlines.open(dst_metadata_file, "w") as writer:
            writer.write(metadata)

        if grid == "2x2":
            # Load image and validate dimensions
            img = Image.open(src_image_file)
            width, height = img.size

            # Validate that both dimensions are even
            if width % 2 != 0 or height % 2 != 0:
                raise ValueError(
                    f"Image dimensions must be even for 2x2 grid. Got {width}x{height}"
                )

            # Calculate crop size (half of each dimension)
            crop_width = width // 2
            crop_height = height // 2

            # Create crop tuples for 4 regions in row-major order
            crop_tuples = [
                (
                    col * crop_width,
                    row * crop_height,
                    col * crop_width + crop_width,
                    row * crop_height + crop_height,
                )
                for row in range(2)
                for col in range(2)
            ]

            # Crop and save each region
            for i, crop_coords in enumerate(crop_tuples):
                crop = img.crop(crop_coords)
                crop.save(dst_samples_dir / f"{i:04d}.png")
        elif grid == "1x1":
            # Default behavior: copy single image to 0000.png
            dst_image_file = dst_samples_dir / "0000.png"
            shutil.copy2(src_image_file, dst_image_file)
        else:
            raise ValueError(f"Invalid grid size: {grid}")

        shutil.copy2(src_sample_dir / "prompt.txt", dst_dir / "prompt.txt")

        matched_count += 1

    print(f"\nConversion complete!")
    print(f"  Matched: {matched_count}")
    print(f"  Unmatched: {unmatched_count}")
    print(f"  Missing files: {missing_files_count}")


@argh.arg("src", help="Source directory path containing sample_XXXXX folders")
@argh.arg("dst", help="Destination directory path for GenEval format output")
@argh.arg("grid", help="Grid size for splitting images", choices=get_args(GridSize))
def main(src: str, dst: str, grid: GridSize) -> None:
    src_path = Path(src)
    dst_path = Path(dst)
    metadata_path = Path("metadata_from_prompt.json")
    with open(metadata_path, "r") as fp:
        metadata_from_prompt = json.load(fp)
    process_samples(src_path, dst_path, metadata_from_prompt, grid=grid)


if __name__ == "__main__":
    argh.dispatch_command(main)
