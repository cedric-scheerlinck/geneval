import argh
from tqdm import tqdm
import jsonlines
import json
import shutil
import sys
from pathlib import Path


def find_matching_metadata(prompt_text: str, metadata_from_prompt: dict[str, tuple[int, dict]]) -> tuple[int, dict] | None:
    for prompt_key, (line_index, metadata) in metadata_from_prompt.items():
        if prompt_key in prompt_text:
            return (line_index, metadata)
    return None

def process_samples(src_path: Path, dst_root: Path, metadata_from_prompt: dict[str, tuple[int, dict]]) -> None:
    dst_root.mkdir(parents=True, exist_ok=True)
    
    matched_count = 0
    unmatched_count = 0
    missing_files_count = 0
    
    src_sample_dirs = sorted(d for d in src_path.iterdir() if d.is_dir() and d.name.startswith('sample_'))
    for src_sample_dir in tqdm(src_sample_dirs, desc="Processing samples"):
        prompt_file = src_sample_dir / 'prompt.txt'
        src_image_file = src_sample_dir / 'image.png'
        try:
            with open(prompt_file, 'r') as fp:
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
        
        dst_samples_dir = dst_dir / 'samples'
        dst_samples_dir.mkdir(parents=True, exist_ok=True)
        
        dst_metadata_file = dst_dir / 'metadata.jsonl'
        with jsonlines.open(dst_metadata_file, 'w') as writer:
            writer.write(metadata)
        
        dst_image_file = dst_samples_dir / '0000.png'
        shutil.copy2(src_image_file, dst_image_file)
        
        matched_count += 1
    
    print(f"\nConversion complete!")
    print(f"  Matched: {matched_count}")
    print(f"  Unmatched: {unmatched_count}")
    print(f"  Missing files: {missing_files_count}")


@argh.arg("src", help="Source directory path containing sample_XXXXX folders")
@argh.arg("dst", help="Destination directory path for GenEval format output")
def main(src: str, dst: str) -> None:
    src_path = Path(src)
    dst_path = Path(dst)
    metadata_path = Path("metadata_from_prompt.json")
    with open(metadata_path, 'r') as fp:
        metadata_from_prompt = json.load(fp)
    process_samples(src_path, dst_path, metadata_from_prompt)


if __name__ == "__main__":
    argh.dispatch_command(main)

