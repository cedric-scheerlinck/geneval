import json
import jsonlines
from pathlib import Path

def main() -> None:
    metadata_path = Path("prompts/evaluation_metadata.jsonl")
    with jsonlines.open(metadata_path) as reader:
        metadata_from_prompt = {line['prompt'].strip(): (i, line) for i, line in enumerate(reader)}

    with open("metadata_from_prompt.json", 'w') as fp:
        json.dump(metadata_from_prompt, fp, indent=2)
    print(f"Saved {len(metadata_from_prompt)} evaluation prompts to metadata_from_prompt.json")

if __name__ == "__main__":
    main()