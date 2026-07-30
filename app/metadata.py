import json
from pathlib import Path

METADATA_PATH = Path("models/metadata.json")

def load_metadata():
    with open(METADATA_PATH, "r") as file:
        return json.load(file)
