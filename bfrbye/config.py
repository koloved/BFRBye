from copy import deepcopy
import yaml
from pathlib import Path

CONFIG_FILE = Path("config.yaml")

default_config = {
    "storage": {
        "methods": ["csv"],   # options: "csv", "txt", "notion"
        "output_file": "output.csv"
    },
    "processing": {
        "interval": 1,         # frames between inference (1 = every frame)
        "mouth_padding": 0.5,   # mouth zone inflation factor
        "trigger_frames": 5,    # consecutive detections to enter ACTIVE state
        "min_duration": 0.5,    # seconds — minimum beep duration per episode
        "cooldown": 0.5,        # seconds — pause after episode before next
        "camera_width": 640,    # capture resolution
        "camera_height": 480
    },
    "notion": {
        "database_id": "",
        "token": ""
    }
}


def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            loaded = yaml.safe_load(f) or {}
        # Migrate old singular "method" -> "methods" on raw loaded data
        storage = loaded.setdefault("storage", {})
        if "method" in storage and "methods" not in storage:
            storage["methods"] = [storage.pop("method")]
        # Fill missing sections/keys from defaults
        for section, dflt in default_config.items():
            if section not in loaded:
                loaded[section] = deepcopy(dflt)
            elif isinstance(dflt, dict):
                for k, v in dflt.items():
                    loaded[section].setdefault(k, deepcopy(v))
        cfg = loaded
    else:
        cfg = deepcopy(default_config)
    return cfg


def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        yaml.safe_dump(config, f)
