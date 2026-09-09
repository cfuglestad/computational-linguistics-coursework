"""01 — Data Preparation

Downloads and preprocesses the SDCNL suicide detection dataset for
replication experiments. Saves processed data to disk.

Dataset: jingjietan/sdcnl-suicide (HuggingFace)
  - Source: Reddit posts from r/SuicideWatch (positive) and r/depression (negative)
  - 1,895 total samples across train/validation/test splits
  - Binary labels: is_suicide (0 = depression/control, 1 = suicidal)
  - Average text length: ~170 words

Domain shift from the original paper:
  Bayram et al. (2022) use transcribed clinical interviews with five
  standardized questions. The SDCNL dataset contains free-form Reddit
  posts, which are longer, less structured, and written rather than
  spoken. This domain shift tests whether the paper's feature extraction
  methods (especially lexical network features) transfer to unstructured
  social media text — a generalizability question the paper raises but
  does not fully answer.

Usage:
    python notebooks/01_data_prep.py
"""

from __future__ import annotations

import io
import json
import os
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

HF_BASE = "https://huggingface.co/datasets/jingjietan/sdcnl-suicide/resolve/main/data"
SPLIT_FILES = {
    "train": "train-00000-of-00001.parquet",
    "validation": "validation-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_splits() -> dict[str, pd.DataFrame]:
    """Download all splits from HuggingFace."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    splits: dict[str, pd.DataFrame] = {}

    for split_name, filename in SPLIT_FILES.items():
        local_path = RAW_DIR / filename

        if local_path.exists():
            print(f"  {split_name}: using cached {local_path}")
            splits[split_name] = pd.read_parquet(local_path)
        else:
            url = f"{HF_BASE}/{filename}"
            print(f"  {split_name}: downloading from {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urllib.request.urlopen(req, timeout=60)
            data = resp.read()

            # Save raw
            local_path.write_bytes(data)

            splits[split_name] = pd.read_parquet(io.BytesIO(data))

        print(f"    {len(splits[split_name])} rows")

    return splits


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean a single Reddit post.

    Minimal preprocessing to match the paper's approach (they use raw
    clinical transcripts with lowercasing). We add Reddit-specific cleanup.
    """
    # Decode HTML entities
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&#x200B;", "")  # zero-width space

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove Reddit formatting artifacts
    text = re.sub(r"\[removed\]", "", text)
    text = re.sub(r"\[deleted\]", "", text)

    # Lowercase (matches Bayram et al.)
    text = text.lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Placeholder tokens that appear in title-only Reddit posts. These are
# artifacts of the scraping process, not real text. 45/46 "emptypost"
# rows are labeled suicidal, making them a label leakage vector.
PLACEHOLDER_TOKENS = {"emptypost", "title", "[removed]", "[deleted]"}

# Minimum word count after cleaning. Very short posts (<=5 words) are
# dominated by placeholders and carry no useful linguistic signal.
MIN_WORD_COUNT = 6


def preprocess_split(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess a single split."""
    out = df[["selftext", "is_suicide"]].copy()
    out.columns = ["text", "label"]

    # Clean text
    out["text"] = out["text"].apply(clean_text)

    # Drop empty texts after cleaning
    out = out[out["text"].str.len() > 0].reset_index(drop=True)

    # Drop placeholder-only posts (label leakage artifact)
    out = out[~out["text"].str.strip().isin(PLACEHOLDER_TOKENS)].reset_index(drop=True)

    # Drop very short texts (mostly artifacts)
    out["_wc"] = out["text"].str.split().str.len()
    out = out[out["_wc"] >= MIN_WORD_COUNT].drop(columns=["_wc"]).reset_index(drop=True)

    # Add word count for analysis
    out["word_count"] = out["text"].str.split().str.len()

    return out


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_stats(splits: dict[str, pd.DataFrame]) -> None:
    """Print dataset summary statistics."""
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)

    for name, df in splits.items():
        n_pos = (df["label"] == 1).sum()
        n_neg = (df["label"] == 0).sum()
        avg_words = df["word_count"].mean()
        med_words = df["word_count"].median()

        print(f"\n{name}:")
        print(f"  Total: {len(df)}")
        print(f"  Suicidal (1): {n_pos} ({100*n_pos/len(df):.1f}%)")
        print(f"  Control  (0): {n_neg} ({100*n_neg/len(df):.1f}%)")
        print(f"  Avg words: {avg_words:.0f}")
        print(f"  Median words: {med_words:.0f}")
        print(f"  Min words: {df['word_count'].min()}")
        print(f"  Max words: {df['word_count'].max()}")

    total = sum(len(df) for df in splits.values())
    print(f"\nTotal across all splits: {total}")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_processed(splits: dict[str, pd.DataFrame]) -> None:
    """Save processed splits as parquet files."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for name, df in splits.items():
        path = PROCESSED_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"  Saved {path} ({len(df)} rows)")

    # Also save a combined version for Monte Carlo CV
    # (The paper merges Corpus 1 and 2 for within-corpus evaluation)
    combined = pd.concat(
        [splits["train"], splits["validation"]], ignore_index=True
    )
    combined_path = PROCESSED_DIR / "train_val_combined.parquet"
    combined.to_parquet(combined_path, index=False)
    print(f"  Saved {combined_path} ({len(combined)} rows, train+val combined)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Step 1: Downloading SDCNL dataset...")
    raw_splits = download_splits()

    print("\nStep 2: Preprocessing...")
    processed = {}
    for name, df in raw_splits.items():
        processed[name] = preprocess_split(df)
        print(f"  {name}: {len(df)} -> {len(processed[name])} rows")

    print_stats(processed)

    print("\nStep 3: Saving processed data...")
    save_processed(processed)

    print("\nDone. Data ready for feature extraction.")


if __name__ == "__main__":
    main()
