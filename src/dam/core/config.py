from pathlib import Path
import os

try:
    import tomllib as toml
except ImportError:
    import tomli as toml

def load_config(path: str = None) -> dict:
    # Default to 'configs/config_score.toml' if not provided
    if path is None:
        path = os.getenv("DAM_CONFIG_PATH", "configs/config_score.toml")
        
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing config file: {p.resolve()}")
        
    with open(p, "rb") as f:
        return toml.load(f)