"""File I/O helpers: read names, chunk-level persistence, cache management."""

import json
from pathlib import Path

import pandas as pd
from libs.gender.llms.config import CHUNK_SEPARATOR


# ── readers ──────────────────────────────────────────────────────────

def read_names(filepath: str, name_var: str) -> list[str]:
    df = pd.read_csv(filepath, keep_default_na=False, na_values=[""])
    df[name_var] = df[name_var].astype(str)
    return df[name_var].dropna().tolist()

def _load_single_file(path: Path, name_var: str) -> dict:
    """Parse a single .json or .csv results file into a dict."""
    ext = path.suffix.lower()
    if ext == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if ext == ".csv":
        df = pd.read_csv(path, keep_default_na=False, na_values=[""])
        df[name_var] = df[name_var].astype(str)
        value_cols = [c for c in df.columns if c != name_var]
        return {
            row[name_var]: {c: row[c] for c in value_cols}
            for _, row in df.iterrows()
        }
    return {}
 


# ── cache ────────────────────────────────────────────────────────────

def load_cache(output_dir: str, prefix: str, ext: str, name_var: str, chunk_separator: str = CHUNK_SEPARATOR) -> dict:
    """Load all results files in `output_dir` matching `prefix*{ext}`.

    Returns a merged dict of all previously queried names.
    """
    directory = Path(output_dir)
    if not directory.is_dir():
        print(f"Cache directory '{output_dir}' does not exist. Starting with empty cache.")
        return {}

    chunk_ids = []
    cache: dict = {}
    for path in sorted(directory.glob(f"{prefix}{chunk_separator}*{ext}")):
        cache.update(_load_single_file(path, name_var))
        cid = path.stem.split(chunk_separator)[-1].split(ext)[0]
        chunk_ids.append(int(cid))
    return cache, max(chunk_ids) if chunk_ids else 0
 


def filter_cached(names: list[str], cache: dict) -> list[str]:
    """Return only names not already present in the cache."""
    print(f"Filtering {len(names)} names against cache of {len(cache)} entries...")
    cached_keys = {k.lower() for k in cache}
    return [n for n in names if n.lower() not in cached_keys]


# ── writers ──────────────────────────────────────────────────────────

def results_to_dataframe(results: dict, name_var: str) -> pd.DataFrame:
    """Convert the nested results dict into a flat DataFrame."""
    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = name_var
    return df.reset_index()


def _write_single(results: dict, path: Path, name_var: str) -> None:
    """Write results to a single file (json or csv based on extension)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext == ".json":
        path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    elif ext == ".csv":
        results_to_dataframe(results, name_var).to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported format '{ext}'. Use .json or .csv")


def write_chunk(results: dict, output_dir: str, fn: str, ext: str, chunk_idx: int, name_var: str, chunk_separator: str) -> Path:
    """Persist a single chunk: {output_dir}/{fn}{chunk_separator}{chunk_idx}{ext}."""
    path = Path(output_dir) / f"{fn}{chunk_separator}{chunk_idx}{ext}"
    _write_single(results, path, name_var=name_var)
    return path


def write_final(results: dict, output_dir: str, fn: str, ext: str, name_var: str) -> Path:
    """Write the consolidated final file: {output_dir}/{fn}{ext}."""
    path = Path(output_dir) / f"{fn}{ext}"
    _write_single(results, path, name_var=name_var)
    return path


def write_text(text: str, output_dir: str, fn: str, ext: str, chunk_idx: int, chunk_separator: str) -> Path:
    """Write raw text to a file (for debugging failed API calls)."""
    path = Path(output_dir) / f"{fn}{chunk_separator}{chunk_idx}{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path