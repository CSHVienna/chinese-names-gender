# From scripts:
# export PYTHONPATH="${PYTHONPATH}:../"

"""
Gender-classify Chinese names (Pinyin) via Claude Opus 4.8.

Usage:
    python main.py names.txt -o results/ -fn gender_results --ext .json

Caching: scans output_dir for files matching {fn}*{ext}, skips already-
queried names. Each API batch is persisted immediately as a chunk file.
A final consolidated file is written once all names are cached.
"""

import argparse
import os
import sys


from libs.gender.llms.config import BATCH_SIZE
from libs.gender.llms.config import CHUNK_SEPARATOR
from libs.gender.llms.config import INFERENCE_MODES

from libs.gender.llms.utils import (
    read_names,
    load_cache,
    filter_cached,
    write_chunk,
    write_final,
    write_text,
)

def batch(items: list, size: int):
    """Yield successive chunks of `size` from `items`."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main() -> None:

    parser = argparse.ArgumentParser(description="Classify gender of Pinyin names via Claude.")
    parser.add_argument("--input_file", help="Text file with one name per line")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    parser.add_argument("--fn", default="gender_results", help="Base filename (no extension)")
    parser.add_argument("--ext", default=".csv", choices=[".json", ".csv"], help="Output format")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--mode", default="categorical", choices=INFERENCE_MODES, help="categorical (label) or numeric (0-1 score)")
    parser.add_argument("--script", default="pinyin", choices=["pinyin", "hanzi"], help="Input script: pinyin or hanzi")
    parser.add_argument("--llm", default="claude", choices=["claude", "gpt"], help="LLM to use: claude or gpt")
    args = parser.parse_args()

    # ── API ───────────────────────────────────────────
    if args.llm == "gpt":
        from openai import OpenAI as APIClient
        from libs.gender.llms.client_openai import classify_batch 
        keyenv = "OPENAI_API_KEY_LEEN"

    elif args.llm == "claude":
        from anthropic import Anthropic as APIClient
        from libs.gender.llms.client_claude import classify_batch 
        keyenv = "ANTHROPIC_API_KEY_LEEN"

    print(f"Using LLM: {args.llm} (API key from {keyenv})")
    key = os.getenv(keyenv)
    if not key:
        print(f"Error: {keyenv} environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    # ── Setup ───────────────────────────────────────────
    if args.script == "pinyin":
        from libs.gender.llms.config_py import MODES
        from libs.gender.llms.config_py import NAME_VAR
    else:
        from libs.gender.llms.config_hz import MODES
        from libs.gender.llms.config_hz import NAME_VAR
    
    client = APIClient(api_key=key)
    print(f"Using mode: {args.mode} ({NAME_VAR} as name variable), and script: {args.script}")
    system_prompt = MODES[args.mode]

    # ── load names & cache ───────────────────────────────────────────
    all_names = read_names(args.input_file, name_var=NAME_VAR)
    cache, max_chunk_idx = load_cache(args.output_dir, prefix=args.fn, ext=args.ext, name_var=NAME_VAR, chunk_separator=CHUNK_SEPARATOR)
    new_names = filter_cached(all_names, cache)

    print(f"Loaded {len(all_names)} names — {len(cache)} cached, {len(new_names)} new")

    if not new_names:
        print("All names already cached — writing final consolidated file.")
        #path = write_final(cache, args.output_dir, args.fn, args.ext)
        #print(f"Final file: {path}")
        return

    # ── query API for new names, persist each chunk ──────────────────
    for i, chunk in enumerate(batch(new_names, args.batch_size), 1):
        i = max_chunk_idx + i  # continue chunk numbering from existing cache
        print(f"  Batch {i}: {len(chunk)} names ...", end=" ", flush=True)
        raw_text, clean, data = classify_batch(chunk, client, system_prompt)

        if data is not None:
            cache.update(data)
        else: 
            print("Warning: API call failed or returned invalid JSON. Skipping this batch.")
            write_text(raw_text, args.output_dir, args.fn, ext=".txt", chunk_idx=i, chunk_separator=CHUNK_SEPARATOR)
            continue

        path = write_chunk(data, args.output_dir, args.fn, args.ext, chunk_idx=i, name_var=NAME_VAR, chunk_separator=CHUNK_SEPARATOR)
        print(f"done → {path}")

    # ── write consolidated final file ────────────────────────────────
    pf = 'gender_label' if args.mode == 'categorical' else "gender_score"
    fn = f"{args.fn}_{pf}"
    path = write_final(cache, args.output_dir, fn, args.ext, name_var=NAME_VAR)
    print(f"Final file: {path} ({len(cache)} total names)")

if __name__ == "__main__":
    main()