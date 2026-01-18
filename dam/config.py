from pathlib import Path

try:
    import tomllib as toml
except ImportError:  # pragma: no cover
    import tomli as toml


def load_config(path: str = "config.toml") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing config: {p.resolve()}")
    with open(p, "rb") as f:
        return toml.load(f)
