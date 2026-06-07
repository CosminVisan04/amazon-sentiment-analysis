"""
Phase 1 — Streaming reservoir sampling.

For each (category × year) cell we draw a uniform random sample of
`samples_per_cell` reviews in a single streaming pass, without ever loading
the full dataset into memory.

Outputs
-------
data/sample_raw.parquet       — sampled reviews
outputs/tables/sampling_summary.{csv,tex}  — population + sample counts per cell
"""
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make `src/` importable when running the script directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.io import load_config, save_parquet, parquet_exists, ensure_dir, setup_logging

log = setup_logging()
logger = logging.getLogger("amazon_sentiment.01_sample")


# ── Reservoir sampling helpers ────────────────────────────────────────────────

def _make_reservoir(k: int) -> dict:
    """Empty reservoir state for a single cell."""
    return {"reservoir": [], "count": 0}


def _update_reservoir(state: dict, item, k: int, rng: random.Random) -> None:
    """
    Algorithm R (Vitter 1985): maintain a uniform random sample of size k
    from a stream without knowing the stream length in advance.
    """
    state["count"] += 1
    n = state["count"]
    if n <= k:
        state["reservoir"].append(item)
    else:
        j = rng.randint(1, n)  # 1-indexed, inclusive
        if j <= k:
            state["reservoir"][j - 1] = item


# ── Record parsing ────────────────────────────────────────────────────────────

def _parse_record(record: dict, category_name: str) -> dict | None:
    """
    Extract and validate a single HuggingFace record.
    Returns None if the record should be skipped.
    """
    text = (record.get("text") or "").strip()
    if not text:
        return None

    ts = record.get("timestamp")
    if ts is None:
        return None
    try:
        # Timestamp is in milliseconds
        dt = datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc)
        year = dt.year
    except (ValueError, OSError):
        return None

    return {
        "category": category_name,
        "year": year,
        "rating": record.get("rating"),
        "title": (record.get("title") or "").strip(),
        "text": text,
        "helpful_vote": record.get("helpful_vote", 0),
        "verified_purchase": record.get("verified_purchase", False),
        "timestamp": int(ts),
    }


# ── Stream source — HuggingFace datasets with JSONL fallback ─────────────────

def _hf_iterator(hf_config: str, cfg: dict):
    """Try load_dataset streaming; raises on failure."""
    from datasets import load_dataset
    ds = load_dataset(
        cfg["hf"]["dataset_name"],
        hf_config,
        split=cfg["hf"]["split"],
        trust_remote_code=cfg["hf"]["trust_remote_code"],
        streaming=True,
    )
    return ds


def _jsonl_iterator(jsonl_name: str, cfg: dict):
    """
    Fallback: stream the JSONL directly from the HuggingFace resolve URL.
    Reads line-by-line — never loads the full file into memory.
    """
    import gzip
    import json
    import urllib.request

    base = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories"
    # Try .jsonl.gz first, then plain .jsonl
    for suffix in (".jsonl.gz", ".jsonl"):
        url = f"{base}/{jsonl_name}{suffix}"
        logger.info(f"JSONL fallback: trying {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
            resp = urllib.request.urlopen(req, timeout=30)
            if suffix == ".jsonl.gz":
                fh = gzip.GzipFile(fileobj=resp)
                for line in fh:
                    yield json.loads(line.decode("utf-8"))
            else:
                for line in resp:
                    yield json.loads(line.decode("utf-8"))
            return
        except Exception as exc:
            logger.warning(f"  {url} failed: {exc}")
    raise RuntimeError(f"Could not retrieve JSONL for {jsonl_name} from HuggingFace.")


def _make_iterator(hf_config: str, category_name: str, cfg: dict):
    """
    Return a tqdm-wrapped record iterator.
    Tries HuggingFace datasets library first; falls back to direct JSONL streaming.
    """
    try:
        from tqdm import tqdm
        ds = _hf_iterator(hf_config, cfg)
        return tqdm(ds, desc=category_name, unit=" reviews", mininterval=5)
    except Exception as exc:
        logger.warning(f"datasets library failed ({exc}); switching to JSONL fallback.")
        # Find jsonl_name from cfg
        jsonl_name = next(
            v["jsonl_name"]
            for v in cfg["categories"].values()
            if v["name"] == category_name
        )
        try:
            from tqdm import tqdm
            return tqdm(_jsonl_iterator(jsonl_name, cfg), desc=category_name, unit=" reviews", mininterval=5)
        except ImportError:
            return _jsonl_iterator(jsonl_name, cfg)


# ── Per-category streaming pass ───────────────────────────────────────────────

def stream_and_sample(
    hf_config: str,
    category_name: str,
    year_start: int,
    year_end: int,
    k: int,
    seed: int,
    cfg: dict,
) -> tuple[dict[int, list[dict]], dict[int, int]]:
    """
    Stream the HuggingFace dataset for one category.

    Returns
    -------
    samples : dict year → list of sampled records
    pop_counts : dict year → total valid population count
    """
    from datasets import load_dataset

    rng = random.Random(seed)

    # One reservoir per year in the target range
    reservoirs: dict[int, dict] = {
        yr: _make_reservoir(k) for yr in range(year_start, year_end + 1)
    }
    pop_counts: dict[int, int] = {yr: 0 for yr in range(year_start, year_end + 1)}

    logger.info(f"Streaming {category_name} ({hf_config}) …")

    iterator = _make_iterator(hf_config, category_name, cfg)

    total_seen = 0
    total_kept = 0

    for record in iterator:
        total_seen += 1
        parsed = _parse_record(record, category_name)
        if parsed is None:
            continue

        yr = parsed["year"]
        if yr not in reservoirs:
            continue

        pop_counts[yr] += 1
        total_kept += 1
        _update_reservoir(reservoirs[yr], parsed, k, rng)

        if total_seen % 500_000 == 0:
            logger.info(
                f"  {category_name}: {total_seen:,} seen, {total_kept:,} in-range"
            )

    logger.info(
        f"  {category_name} done: {total_seen:,} total records seen, "
        f"{total_kept:,} valid in {year_start}–{year_end}"
    )

    samples = {yr: state["reservoir"] for yr, state in reservoirs.items()}
    return samples, pop_counts


# ── Sampling summary table ────────────────────────────────────────────────────

def _build_summary(
    all_samples: dict[str, dict[int, list]],
    all_pops: dict[str, dict[int, int]],
    k: int,
    enforce_equal: bool,
) -> "pd.DataFrame":
    import pandas as pd

    rows = []
    for cat, year_samples in all_samples.items():
        for yr, records in sorted(year_samples.items()):
            pop = all_pops[cat][yr]
            sampled = len(records)
            shortfall = max(0, k - pop)
            rows.append({
                "category": cat,
                "year": yr,
                "population_count": pop,
                "sampled_count": sampled,
                "target_count": k,
                "shortfall": shortfall,
            })

    return pd.DataFrame(rows)


def _enforce_equal_cells(
    all_samples: dict[str, dict[int, list]],
    seed: int,
) -> dict[str, dict[int, list]]:
    """Down-cap every cell to the global minimum cell size."""
    import random as _random

    rng = _random.Random(seed)
    min_size = min(
        len(records)
        for year_samples in all_samples.values()
        for records in year_samples.values()
    )
    logger.info(f"enforce_equal_cells: down-capping all cells to {min_size}")

    trimmed: dict[str, dict[int, list]] = {}
    for cat, year_samples in all_samples.items():
        trimmed[cat] = {}
        for yr, records in year_samples.items():
            if len(records) > min_size:
                trimmed[cat][yr] = rng.sample(records, min_size)
            else:
                trimmed[cat][yr] = records
    return trimmed


# ── LaTeX table renderer ──────────────────────────────────────────────────────

def _to_latex(df: "pd.DataFrame") -> str:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Sampling summary: population counts and sampled counts per (category $\times$ year) cell.}",
        r"\label{tab:sampling_summary}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Category & Year & Population & Sampled & Target & Shortfall \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['category']} & {int(row['year'])} & "
            f"{int(row['population_count']):,} & {int(row['sampled_count']):,} & "
            f"{int(row['target_count']):,} & {int(row['shortfall']):,} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()

    raw_out = "data/sample_raw.parquet"
    if parquet_exists(raw_out):
        logger.info(f"Cache hit: {raw_out} already exists. Delete it to re-run Phase 1.")
        return

    year_start = cfg["year_range"]["start"]
    year_end = cfg["year_range"]["end"]
    k = cfg["sampling"]["samples_per_cell"]
    seed = cfg["sampling"]["seed"]
    enforce_equal = cfg["sampling"]["enforce_equal_cells"]

    all_samples: dict[str, dict[int, list]] = {}
    all_pops: dict[str, dict[int, int]] = {}

    for cat_key, cat_cfg in cfg["categories"].items():
        cat_name = cat_cfg["name"]
        hf_config = cat_cfg["hf_config"]
        samples, pops = stream_and_sample(
            hf_config=hf_config,
            category_name=cat_name,
            year_start=year_start,
            year_end=year_end,
            k=k,
            seed=seed,
            cfg=cfg,
        )
        all_samples[cat_name] = samples
        all_pops[cat_name] = pops

    if enforce_equal:
        all_samples = _enforce_equal_cells(all_samples, seed)

    # Flatten to a DataFrame
    import pandas as pd
    rows = [
        record
        for year_samples in all_samples.values()
        for records in year_samples.values()
        for record in records
    ]
    df = pd.DataFrame(rows)
    logger.info(f"Total sampled rows: {len(df):,}")

    save_parquet(df, raw_out)
    logger.info(f"Saved → {raw_out}")

    # Sampling summary
    summary = _build_summary(all_samples, all_pops, k, enforce_equal)
    tables_dir = cfg["paths"]["tables_dir"]
    ensure_dir(tables_dir)

    from pathlib import Path
    from utils.io import project_root

    csv_path = project_root() / tables_dir / "sampling_summary.csv"
    tex_path = project_root() / tables_dir / "sampling_summary.tex"

    summary.to_csv(csv_path, index=False)
    tex_path.write_text(_to_latex(summary), encoding="utf-8")

    logger.info(f"Sampling summary → {csv_path}")
    logger.info(f"Sampling summary (LaTeX) → {tex_path}")

    # Print a compact preview
    print("\n=== Sampling Summary ===")
    print(summary.to_string(index=False))

    # Log shortfalls
    shortfalls = summary[summary["shortfall"] > 0]
    if not shortfalls.empty:
        logger.warning(
            f"{len(shortfalls)} cells had fewer than {k} reviews:\n"
            + shortfalls[["category", "year", "population_count"]].to_string(index=False)
        )
    else:
        logger.info("No shortfall cells — all cells met the target sample size.")


if __name__ == "__main__":
    main()
