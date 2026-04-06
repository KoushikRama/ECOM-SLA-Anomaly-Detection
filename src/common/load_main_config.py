from pathlib import Path
import json
import os

def get_root_dir():
    return Path(__file__).resolve().parent.parent.parent


def load_main_config():
    BASE_DIR = get_root_dir()
    config_path = BASE_DIR / "config" / "main.config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Main config not found: {config_path}")

    with open(config_path, "r") as f:
        return json.load(f)


def get_path(key, config):
    return get_root_dir() / config["paths"][key]


def load_data_config():
    main_config = load_main_config()
    data_config_path = get_path("data_config", main_config)

    if not Path(data_config_path).exists():
        raise FileNotFoundError(f"Data config not found: {data_config_path}")

    with open(data_config_path, "r") as f:
        return json.load(f)
    
def get_data_filepath(config=None):
    if config is None:
        config = load_main_config()
    filename = config["data"]["filename"]
    filepath = os.path.join(get_root_dir(), "data" ,"raw", filename)
    return filepath

def get_model_path(config=None):
    if config is None:
        config = load_main_config()
    model = config["paths"]["model"]
    modelpath = os.path.join(get_root_dir(), model)
    return Path(modelpath)

def get_active_version(config=None):
    if config is None:
        config = load_main_config()
    return config["active_version"]
       