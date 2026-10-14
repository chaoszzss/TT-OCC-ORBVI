"""An explicit named garage class registry; 255 always means unknown semantics."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / "taxonomy/garage.json"
UNKNOWN = 255


def load_registry(path=DEFAULT_REGISTRY):
    registry = json.loads(Path(path).read_text())
    ids = [int(c["id"]) for c in registry["classes"]]
    if ids != list(range(len(ids))) or registry["unknown_id"] != UNKNOWN:
        raise ValueError("Garage registry must have contiguous known IDs and unknown=255")
    return registry


def class_mapping(source_names, registry=None):
    """Map ADE20K or a custom OpenSeeD vocabulary by names, never by raw IDs."""
    registry = registry or load_registry()
    target = {c["name"]: c["id"] for c in registry["classes"]}
    aliases = {
        "wall":"wall", "building":"wall", "floor":"floor", "ceiling":"ceiling",
        "column":"column", "pillar":"column", "car":"car", "truck":"car",
        "bus":"car", "van":"car", "vehicle":"car", "person":"person",
        "pedestrian":"person", "door":"door", "doorframe":"door", "road":"road",
        "tree":"vegetation", "grass":"vegetation", "plant":"vegetation", "sky":"sky",
    }
    result = {}
    for key, name in source_names.items():
        tokens = [t.strip() for t in str(name).lower().replace(";", ",").split(",")]
        name = next((aliases[t] for t in tokens if t in aliases), "other")
        result[int(key)] = target[name]
    return result


def palette(registry=None):
    registry = registry or load_registry()
    colors = np.tile(np.array(registry["unknown"]["color"], np.uint8), (256, 1))
    for item in registry["classes"]:
        colors[item["id"]] = item["color"]
    return colors


def class_histogram(labels, registry=None):
    registry = registry or load_registry()
    names = {c["id"]: c["name"] for c in registry["classes"]}
    names[UNKNOWN] = "unknown"
    ids, counts = np.unique(np.asarray(labels), return_counts=True)
    return {names.get(int(i), f"invalid_{i}"): int(n) for i, n in zip(ids, counts)}
