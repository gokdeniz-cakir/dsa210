"""
Corpus loading and processing utilities.

Provides functions for loading musical scores from various sources
and processing them through the feature extraction pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from tqdm import tqdm

if TYPE_CHECKING:
    from music21.stream import Score

logger = logging.getLogger(__name__)


def load_music21_corpus(
    composer: str = "bach",
    limit: int | None = None,
) -> Iterator[tuple[str, "Score"]]:
    """
    Load pieces from music21's built-in corpus.

    Parameters
    ----------
    composer : str, optional
        Composer name to filter by (default "bach")
    limit : int | None, optional
        Maximum number of pieces to load

    Yields
    ------
    tuple[str, Score]
        Tuple of (piece_id, Score object)

    Examples
    --------
    >>> for piece_id, score in load_music21_corpus("bach", limit=10):
    ...     features = extract_features(score)
    """
    from music21 import corpus

    # Get all paths for the composer
    paths = corpus.getComposer(composer)

    if limit:
        paths = paths[:limit]

    for path in paths:
        piece_id = str(path).split("/")[-1] if "/" in str(path) else str(path)
        try:
            score = corpus.parse(path)
            yield piece_id, score
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            continue


def load_from_directory(
    directory: Path | str,
    extensions: tuple[str, ...] = (".mxl", ".musicxml", ".xml", ".mid", ".midi", ".krn"),
    limit: int | None = None,
) -> Iterator[tuple[str, "Score"]]:
    """
    Load pieces from a local directory.

    Parameters
    ----------
    directory : Path | str
        Directory containing score files
    extensions : tuple[str, ...], optional
        File extensions to include
    limit : int | None, optional
        Maximum number of pieces to load

    Yields
    ------
    tuple[str, Score]
        Tuple of (filename, Score object)
    """
    from music21 import converter

    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = [f for f in directory.iterdir() if f.suffix.lower() in extensions]

    if limit:
        files = files[:limit]

    for filepath in files:
        try:
            score = converter.parse(str(filepath))
            yield filepath.name, score
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")
            continue


def extract_corpus_features(
    corpus_loader: Iterator[tuple[str, "Score"]],
    n_workers: int | None = None,
    show_progress: bool = True,
) -> list[dict]:
    """
    Extract features from all pieces in a corpus.

    Parameters
    ----------
    corpus_loader : Iterator[tuple[str, Score]]
        Iterator yielding (piece_id, Score) tuples
    n_workers : int | None, optional
        Number of parallel workers (currently unused, sequential for safety)
    show_progress : bool, optional
        Whether to show progress bar

    Returns
    -------
    list[dict]
        List of feature dictionaries with 'piece_id' added
    """
    from .extractors import extract_features

    results = []
    pieces = list(corpus_loader)  # Materialize for progress bar

    iterator = tqdm(pieces, desc="Extracting features") if show_progress else pieces

    for piece_id, score in iterator:
        try:
            features = extract_features(score)
            if features:
                features["piece_id"] = piece_id
                results.append(features)
        except Exception as e:
            logger.warning(f"Failed to extract features from {piece_id}: {e}")
            continue

    return results


def features_to_dataframe(features: list[dict]) -> "pd.DataFrame":
    """
    Convert feature list to pandas DataFrame.

    Parameters
    ----------
    features : list[dict]
        List of feature dictionaries

    Returns
    -------
    pd.DataFrame
        DataFrame with piece_id as first column
    """
    import pandas as pd

    df = pd.DataFrame(features)

    # Reorder columns to put piece_id first
    if "piece_id" in df.columns:
        cols = ["piece_id"] + [c for c in df.columns if c != "piece_id"]
        df = df[cols]

    return df
