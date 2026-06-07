"""
I/O helpers: config loading, parquet read/write, path resolution, logging setup.
"""
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_root() -> Path:
    return _PROJECT_ROOT


def load_config(path: str | Path | None = None) -> dict:
    if path is None:
        path = _PROJECT_ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str, cfg: dict | None = None) -> Path:
    return _PROJECT_ROOT / relative


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_parquet(df: pd.DataFrame, relative_path: str) -> Path:
    out = _PROJECT_ROOT / relative_path
    ensure_dir(out.parent)
    df.to_parquet(out, index=False)
    return out


def load_parquet(relative_path: str) -> pd.DataFrame:
    path = _PROJECT_ROOT / relative_path
    return pd.read_parquet(path)


def parquet_exists(relative_path: str) -> bool:
    return (_PROJECT_ROOT / relative_path).exists()


def setup_logging(log_file: str = "run.log", level: int = logging.INFO) -> logging.Logger:
    log_path = _PROJECT_ROOT / log_file
    ensure_dir(log_path.parent)

    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)
    # Suppress noisy third-party loggers
    for noisy in ("fontTools", "httpx", "huggingface_hub", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("amazon_sentiment")
