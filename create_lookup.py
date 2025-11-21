import json
import jsonlines
from pathlib import Path

def main() -> None:
    metadata_path = Path("prompts/evaluation_metadata.jsonl")
    metadata_from_prompt = {}
    with jsonlines.open(metadata_path) as reader:
        for i, line in enumerate(reader):
            prompt_key = line['prompt'].strip()
            if prompt_key in metadata_from_prompt:
                raise ValueError(f"Duplicate prompt found: {prompt_key}")
            metadata_from_prompt[prompt_key] = (i, line)

    with open("metadata_from_prompt.json", 'w') as fp:
        json.dump(metadata_from_prompt, fp, indent=2)
    print(f"Saved {len(metadata_from_prompt)} evaluation prompts to metadata_from_prompt.json")