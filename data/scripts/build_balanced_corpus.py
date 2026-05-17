#!/usr/bin/env python3
"""
Build a balanced corpus by selecting the most popular works per composer.

Workflow:
1. Process all MIDI files from specified composers
2. Collect Spotify popularity for each piece
3. Keep only top N most popular per composer

Usage:
    python scripts/build_balanced_corpus.py \
        --composers bach beethoven mozart chopin \
        --top-per-composer 100 \
        --output data/features/balanced_corpus.csv
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from huggingface_hub import hf_hub_download, list_repo_files
from music21 import converter
from tqdm import tqdm
import pandas as pd

from src.feature_extraction import extract_features
from src.data_collection.spotify import (
    SpotifyCollector,
    load_credentials_from_env,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

REPO_ID = "drengskapur/midi-classical-music"


def parse_composer(filename: str) -> str:
    """Extract composer name from filename."""
    name = filename.replace("data/", "")
    composer = name.split("-")[0] if "-" in name else "unknown"
    return composer.lower()


def get_piece_title(filename: str) -> str:
    """Extract piece title for Spotify search."""
    name = filename.replace("data/", "").replace(".mid", "")
    parts = name.split("-", 1)
    if len(parts) > 1:
        return parts[1].replace("_", " ").replace("  ", " ").strip()
    return name


def process_midi_file(filepath: str) -> dict | None:
    """Process MIDI file and extract features."""
    try:
        score = converter.parse(filepath)
        return extract_features(score)
    except Exception as e:
        logger.debug(f"Failed to parse: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Build balanced corpus with most popular works per composer"
    )
    parser.add_argument(
        "--composers",
        type=str,
        nargs="+",
        required=True,
        help="Composers to include",
    )
    parser.add_argument(
        "--top-per-composer",
        type=int,
        default=100,
        help="Keep top N most popular per composer (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/features/balanced_corpus.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip feature extraction (just get popularity to preview)",
    )
    args = parser.parse_args()

    # Load Spotify credentials
    logger.info("Loading Spotify credentials...")
    client_id, client_secret = load_credentials_from_env()
    collector = SpotifyCollector(client_id, client_secret)

    # List all files
    logger.info("Listing files in Hugging Face repo...")
    all_files = list_repo_files(REPO_ID, repo_type="dataset")
    midi_files = [f for f in all_files if f.endswith(".mid")]

    # Filter by composers
    composers_lower = [c.lower() for c in args.composers]
    midi_files = [f for f in midi_files if parse_composer(f) in composers_lower]
    logger.info(f"Found {len(midi_files)} files for {len(composers_lower)} composers")

    # Group by composer
    by_composer = defaultdict(list)
    for f in midi_files:
        by_composer[parse_composer(f)].append(f)

    # Show counts
    print("\nFiles per composer:")
    for c in sorted(by_composer.keys()):
        print(f"  {c:<20} {len(by_composer[c]):>4} files")

    # Step 1: Get Spotify popularity for all pieces
    print("\n" + "=" * 60)
    print("STEP 1: Getting Spotify popularity for all pieces")
    print("=" * 60)

    popularity_data = []
    for composer in tqdm(composers_lower, desc="Composers"):
        files = by_composer.get(composer, [])
        for filename in tqdm(files, desc=f"  {composer}", leave=False):
            piece_id = filename.replace("data/", "").replace(".mid", "")
            title = get_piece_title(filename)
            query = f"{composer} {title}"

            try:
                tracks = collector.search_composition(query, limit=20)
                agg = collector.aggregate_play_counts(tracks, method="max")
                popularity = agg["popularity_max"]
            except Exception as e:
                logger.debug(f"Spotify search failed for {piece_id}: {e}")
                popularity = 0

            popularity_data.append({
                "filename": filename,
                "piece_id": piece_id,
                "composer": composer,
                "title": title,
                "popularity": popularity,
            })

    pop_df = pd.DataFrame(popularity_data)

    # Step 2: Select top N per composer
    print("\n" + "=" * 60)
    print(f"STEP 2: Selecting top {args.top_per_composer} per composer")
    print("=" * 60)

    selected = []
    for composer in composers_lower:
        composer_df = pop_df[pop_df["composer"] == composer]
        top_n = composer_df.nlargest(args.top_per_composer, "popularity")
        selected.append(top_n)
        print(f"  {composer:<20} selected {len(top_n):>3} pieces "
              f"(popularity range: {top_n['popularity'].min():.0f}-{top_n['popularity'].max():.0f})")

    selected_df = pd.concat(selected, ignore_index=True)
    print(f"\nTotal selected: {len(selected_df)} pieces")

    if args.skip_features:
        # Just save popularity data for preview
        selected_df.to_csv(args.output.with_suffix(".popularity.csv"), index=False)
        print(f"\nSaved popularity preview to {args.output.with_suffix('.popularity.csv')}")
        print("Run without --skip-features to extract features")
        return

    # Step 3: Extract features for selected pieces
    print("\n" + "=" * 60)
    print("STEP 3: Extracting features for selected pieces")
    print("=" * 60)

    results = []
    failed = 0

    for _, row in tqdm(selected_df.iterrows(), total=len(selected_df), desc="Extracting"):
        try:
            filepath = hf_hub_download(
                repo_id=REPO_ID,
                filename=row["filename"],
                repo_type="dataset",
            )
            features = process_midi_file(filepath)
            if features:
                features["piece_id"] = row["piece_id"]
                features["composer"] = row["composer"]
                features["title"] = row["title"]
                features["popularity"] = row["popularity"]
                results.append(features)
            else:
                failed += 1
        except Exception as e:
            logger.debug(f"Failed: {row['piece_id']}: {e}")
            failed += 1

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)

    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total pieces: {len(df)}")
    print(f"Failed: {failed}")
    print(f"Output: {args.output}")

    print("\nBy composer:")
    for composer in df["composer"].unique():
        subset = df[df["composer"] == composer]
        print(f"  {composer:<20} {len(subset):>3} pieces, "
              f"popularity: {subset['popularity'].mean():.1f} avg")

    print("=" * 60)


if __name__ == "__main__":
    main()
