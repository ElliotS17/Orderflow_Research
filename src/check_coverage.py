from pathlib import Path
import json
import warnings

warnings.filterwarnings(
    "ignore",
    category=Warning,
    module=r"databento\.common\.dbnstore"
)

raw_dir = Path("data/raw")

with open(raw_dir / "manifest.json") as f:
    manifest = json.load(f)

with open(raw_dir / "metadata.json") as f:
    metadata = json.load(f)

expected = {
    item["filename"]
    for item in manifest["files"]
    if item["filename"].endswith(".dbn.zst")
}

actual = {
    path.name
    for path in raw_dir.glob("*.dbn.zst")
}

print("===== MANIFEST VS LOCAL FILES =====")
print("Expected DBN files:", len(expected))
print("Local DBN files:", len(actual))
print("Missing locally:", sorted(expected - actual))
print("Unexpected locally:", sorted(actual - expected))

print("\n===== FULL DATASET METADATA =====")
print(json.dumps(metadata, indent=2))

print("\n===== MANIFEST DATE FILES =====")
for filename in sorted(expected):
    print(filename)