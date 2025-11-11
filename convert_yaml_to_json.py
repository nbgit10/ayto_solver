"""Convert YAML example files to JSON format."""
import yaml
import json
from pathlib import Path


def convert_yaml_to_json(yaml_file: Path, json_file: Path):
    """Convert a single YAML file to JSON."""
    with open(yaml_file, 'r', encoding='utf-8') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    # Convert to API format
    json_data = {
        "males": data["MALES"],
        "females": data["FEMALES"],
        "matching_nights": [
            {
                "pairs": night["Pairs"],
                "matches": night["Matches"]
            }
            for night in data["MATCHING_NIGHTS"]
        ],
        "truth_booths": [
            {
                "pair": tb["Pair"],
                "match": tb["Match"]
            }
            for tb in data["TRUTH_BOOTH"]
        ]
    }

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"Converted {yaml_file.name} -> {json_file.name}")


def main():
    """Convert all YAML examples to JSON."""
    examples_dir = Path("examples")
    json_dir = examples_dir / "json"
    json_dir.mkdir(exist_ok=True)

    yaml_files = list(examples_dir.glob("*.yaml"))

    for yaml_file in yaml_files:
        json_file = json_dir / f"{yaml_file.stem}.json"
        convert_yaml_to_json(yaml_file, json_file)

    print(f"\nConverted {len(yaml_files)} files to JSON format in {json_dir}")


if __name__ == "__main__":
    main()
