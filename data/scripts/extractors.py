"""
Feature extraction functions for classical music scores.

This module provides the core feature extraction functionality using music21.
Features are based on the methodology described in paper.
Structural features include approximate harmonic rhythm, cadence density,
windowed key movement, and sectional repetition signals.

Note: This is a non-normative benchmark. Features measure observable properties,
not music "quality" or "goodness."
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from multiprocessing import Pool, cpu_count
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from music21.stream import Score


_KRUMHANSL_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=float,
)
_KRUMHANSL_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=float,
)
_KRUMHANSL_MAJOR /= _KRUMHANSL_MAJOR.sum()
_KRUMHANSL_MINOR /= _KRUMHANSL_MINOR.sum()


def _choose_window_size(note_count: int) -> int:
    if note_count <= 0:
        return 0
    return max(16, min(64, note_count // 10))


def _estimate_key_from_hist(hist: np.ndarray) -> tuple[int, str] | None:
    if hist.sum() <= 0:
        return None
    hist = hist / hist.sum()
    best_score = -1.0
    best_key: tuple[int, str] | None = None
    for tonic in range(12):
        major_score = float(np.dot(hist, np.roll(_KRUMHANSL_MAJOR, tonic)))
        minor_score = float(np.dot(hist, np.roll(_KRUMHANSL_MINOR, tonic)))
        if major_score > best_score:
            best_score = major_score
            best_key = (tonic, "major")
        if minor_score > best_score:
            best_score = minor_score
            best_key = (tonic, "minor")
    return best_key


def _estimate_key_from_pitch_classes(pitch_classes: list[int]) -> tuple[int, str] | None:
    if not pitch_classes:
        return None
    hist = np.bincount(pitch_classes, minlength=12).astype(float)
    return _estimate_key_from_hist(hist)


def _pc_distance(pc_a: int, pc_b: int) -> int:
    diff = (pc_a - pc_b) % 12
    return min(diff, 12 - diff)


def _compute_window_features(pitch_classes: list[int]) -> dict[str, float]:
    defaults = {
        "key_change_rate": 0.0,
        "modulation_mean": 0.0,
        "sectional_repetition_ratio": 0.0,
        "sectional_similarity_mean": 0.0,
    }
    if len(pitch_classes) < 16:
        return defaults

    window_size = _choose_window_size(len(pitch_classes))
    if window_size <= 0:
        return defaults
    min_window = max(8, window_size // 2)

    hists: list[np.ndarray] = []
    for start in range(0, len(pitch_classes), window_size):
        window = pitch_classes[start : start + window_size]
        if len(window) < min_window:
            continue
        hist = np.bincount(window, minlength=12).astype(float)
        if hist.sum() <= 0:
            continue
        hists.append(hist)

    if len(hists) < 2:
        return defaults

    # Key changes over windows
    keys: list[tuple[int, str]] = []
    for hist in hists:
        key = _estimate_key_from_hist(hist)
        if key:
            keys.append(key)

    key_change_rate = 0.0
    modulation_mean = 0.0
    if len(keys) >= 2:
        changes = sum(1 for a, b in zip(keys, keys[1:]) if a != b)
        key_change_rate = changes / (len(keys) - 1)

        distances = [_pc_distance(a[0], b[0]) for a, b in zip(keys, keys[1:])]
        modulation_mean = statistics.mean(distances) if distances else 0.0

    # Sectional repetition from window pitch-class histograms
    vectors = []
    for hist in hists:
        norm = np.linalg.norm(hist)
        if norm == 0:
            continue
        vectors.append(hist / norm)

    sectional_repetition_ratio = 0.0
    sectional_similarity_mean = 0.0
    if len(vectors) >= 2:
        mat = np.vstack(vectors)
        sims = mat @ mat.T
        max_sims = []
        for i in range(len(vectors)):
            row = np.delete(sims[i], i)
            max_sims.append(float(np.max(row)))
        sectional_similarity_mean = statistics.mean(max_sims) if max_sims else 0.0
        sectional_repetition_ratio = (
            sum(1 for s in max_sims if s >= 0.85) / len(max_sims)
            if max_sims
            else 0.0
        )

    return {
        "key_change_rate": float(key_change_rate),
        "modulation_mean": float(modulation_mean),
        "sectional_repetition_ratio": float(sectional_repetition_ratio),
        "sectional_similarity_mean": float(sectional_similarity_mean),
    }


def _compute_chord_features(
    piece: "Score",
    key_tonic: int | None,
) -> dict[str, float]:
    from music21 import chord

    defaults = {
        "chord_change_rate": 0.0,
        "chord_change_std": 0.0,
        "avg_chord_size": 0.0,
        "avg_vertical_range": 0.0,
        "chord_pc_entropy": 0.0,
        "tonic_dominant_ratio": 0.0,
        "cadence_rate": 0.0,
    }

    try:
        chordified = piece.chordify()
    except Exception:
        return defaults

    chords = [
        c for c in chordified.recurse().getElementsByClass(chord.Chord) if c.pitches
    ]
    if not chords:
        return defaults

    total_duration = float(piece.duration.quarterLength)
    pairs = sorted(((float(c.offset), c) for c in chords), key=lambda x: x[0])

    segments = []
    for idx, (offset, chord_obj) in enumerate(pairs):
        next_offset = pairs[idx + 1][0] if idx + 1 < len(pairs) else total_duration
        duration = max(0.0, next_offset - offset)
        pcs = tuple(sorted({p.pitchClass for p in chord_obj.pitches}))
        if not pcs:
            continue
        if segments and pcs == segments[-1]["pcs"]:
            segments[-1]["duration"] += duration
            continue
        segments.append({"pcs": pcs, "duration": duration, "chord": chord_obj})

    if not segments:
        return defaults

    segment_durations = [s["duration"] for s in segments if s["duration"] > 0]
    total_weight = sum(segment_durations)

    chord_change_rate = (
        (len(segments) - 1) / total_duration if total_duration > 0 else 0.0
    )
    chord_change_std = (
        statistics.stdev(segment_durations) if len(segment_durations) > 1 else 0.0
    )

    avg_chord_size = 0.0
    avg_vertical_range = 0.0
    if total_weight > 0:
        size_sum = sum(len(s["pcs"]) * s["duration"] for s in segments)
        avg_chord_size = size_sum / total_weight
        range_sum = 0.0
        for s in segments:
            pitches = [p.midi for p in s["chord"].pitches]
            if pitches:
                range_sum += (max(pitches) - min(pitches)) * s["duration"]
        avg_vertical_range = range_sum / total_weight if total_weight > 0 else 0.0

    pc_counts = Counter()
    for s in segments:
        pc_counts[s["pcs"]] += s["duration"]
    pc_total = sum(pc_counts.values())
    if pc_total > 0 and len(pc_counts) > 1:
        nonzero_counts = [c for c in pc_counts.values() if c > 0]
        if nonzero_counts and len(nonzero_counts) > 1:
            pc_entropy = -sum(
                (c / pc_total) * math.log2(c / pc_total) for c in nonzero_counts
            )
            chord_pc_entropy = pc_entropy / math.log2(len(nonzero_counts))
        else:
            chord_pc_entropy = 0.0
    else:
        chord_pc_entropy = 0.0

    tonic_dominant_ratio = 0.0
    cadence_rate = 0.0
    if key_tonic is not None:
        root_intervals: list[int] = []
        for s in segments:
            root = s["chord"].root()
            if root is None:
                continue
            root_intervals.append((root.pitchClass - key_tonic) % 12)

        if root_intervals:
            tonic_dominant_ratio = sum(
                1 for i in root_intervals if i in (0, 7)
            ) / len(root_intervals)

        if len(root_intervals) >= 2:
            cadence_count = sum(
                1
                for prev, curr in zip(root_intervals, root_intervals[1:])
                if prev == 7 and curr == 0
            )
            cadence_rate = cadence_count / (len(root_intervals) - 1)

    return {
        "chord_change_rate": float(chord_change_rate),
        "chord_change_std": float(chord_change_std),
        "avg_chord_size": float(avg_chord_size),
        "avg_vertical_range": float(avg_vertical_range),
        "chord_pc_entropy": float(chord_pc_entropy),
        "tonic_dominant_ratio": float(tonic_dominant_ratio),
        "cadence_rate": float(cadence_rate),
    }


def extract_features(piece: Score) -> dict[str, Any] | None:
    """
    Extract computational features from a music21 score.

    Extracts pitch, interval, rhythm, contour, repetition, and structural features
    from a musical score. Returns None if the piece has fewer than 10 notes.

    Parameters
    ----------
    piece : music21.stream.Score
        A music21 Score object to analyze

    Returns
    -------
    dict[str, Any] | None
        Dictionary of extracted features, or None if piece is too short.
        Features include:
        - total_notes: Total number of notes
        - duration: Total duration in quarter lengths
        - pitch_range: Difference between highest and lowest pitch (semitones)
        - pitch_mean: Mean MIDI pitch value
        - pitch_std: Standard deviation of pitches
        - pitch_class_entropy: Normalized entropy of pitch class distribution
        - interval_mean: Mean interval size (semitones)
        - stepwise_ratio: Proportion of intervals <= 2 semitones
        - leap_ratio: Proportion of intervals > 4 semitones
        - unique_durations: Count of unique note durations
        - duration_std: Standard deviation of note durations
        - note_density: Notes per quarter length
        - contour_change_rate: Rate of melodic direction changes
        - repetition_ratio: Proportion of repeated 4-note patterns
        - chord_change_rate: Chord changes per quarter length
        - chord_change_std: Std dev of chord segment durations
        - avg_chord_size: Average pitch classes per chord segment
        - avg_vertical_range: Average vertical pitch span per chord segment
        - chord_pc_entropy: Normalized entropy of chord pitch-class sets
        - tonic_dominant_ratio: Share of chords on tonic or dominant
        - cadence_rate: Share of dominant-to-tonic transitions
        - key_change_rate: Windowed key-change rate
        - modulation_mean: Mean semitone distance between windowed keys
        - sectional_repetition_ratio: Share of windows with a close repeat
        - sectional_similarity_mean: Mean max window similarity

    References
    ----------
    Based on proof-of-concept in classical_music_benchmark_project.md (lines 128-193)
    """
    flat = piece.flatten()
    notes = [n for n in flat.notes if hasattr(n, "pitch")]

    if len(notes) < 10:
        return None

    features: dict[str, Any] = {}

    # Basic metrics
    features["total_notes"] = len(notes)
    features["duration"] = piece.duration.quarterLength

    # Pitch features
    pitches = [n.pitch.midi for n in notes]
    pitch_classes = [p % 12 for p in pitches]
    features["pitch_range"] = max(pitches) - min(pitches)
    features["pitch_mean"] = statistics.mean(pitches)
    features["pitch_std"] = statistics.stdev(pitches) if len(pitches) > 1 else 0.0

    # Pitch class entropy (normalized)
    pc_counts = Counter(pitch_classes)
    total = sum(pc_counts.values())
    entropy = -sum((c / total) * math.log2(c / total) for c in pc_counts.values())
    features["pitch_class_entropy"] = entropy / 3.585  # Normalize to [0, 1]

    # Melodic interval features (computed per part to preserve voice leading)
    intervals: list[int] = []
    for part in piece.parts:
        part_notes = [n for n in part.flatten().notes if hasattr(n, "pitch")]
        for i in range(1, len(part_notes)):
            intervals.append(
                abs(part_notes[i].pitch.midi - part_notes[i - 1].pitch.midi)
            )

    if intervals:
        features["interval_mean"] = statistics.mean(intervals)
        features["stepwise_ratio"] = sum(1 for i in intervals if i <= 2) / len(
            intervals
        )
        features["leap_ratio"] = sum(1 for i in intervals if i > 4) / len(intervals)
    else:
        features["interval_mean"] = 0.0
        features["stepwise_ratio"] = 0.0
        features["leap_ratio"] = 0.0

    # Rhythm features
    durations = [n.duration.quarterLength for n in notes]
    features["unique_durations"] = len(set(durations))
    features["duration_std"] = (
        statistics.stdev(durations) if len(durations) > 1 else 0.0
    )

    # Note density
    if piece.duration.quarterLength > 0:
        features["note_density"] = len(notes) / piece.duration.quarterLength
    else:
        features["note_density"] = 0.0

    # Contour features (direction change rate)
    changes = sum(
        1
        for i in range(2, len(pitches))
        if (pitches[i - 1] - pitches[i - 2]) * (pitches[i] - pitches[i - 1]) < 0
    )
    features["contour_change_rate"] = changes / len(pitches) if pitches else 0.0

    # Repetition features (4-gram repetition ratio)
    four_grams = [tuple(pitches[i : i + 4]) for i in range(len(pitches) - 3)]
    if four_grams:
        gram_counts = Counter(four_grams)
        features["repetition_ratio"] = (
            sum(1 for c in gram_counts.values() if c > 1) / len(gram_counts)
        )
    else:
        features["repetition_ratio"] = 0.0

    key_info = _estimate_key_from_pitch_classes(pitch_classes)
    key_tonic = key_info[0] if key_info else None

    features.update(_compute_chord_features(piece, key_tonic))
    features.update(_compute_window_features(pitch_classes))

    return features


def extract_features_batch(
    pieces: list[Score], n_workers: int | None = None
) -> list[dict[str, Any] | None]:
    """
    Extract features from multiple scores in parallel.

    Uses multiprocessing to efficiently process large corpora.

    Parameters
    ----------
    pieces : list[Score]
        List of music21 Score objects to analyze
    n_workers : int | None, optional
        Number of worker processes. Defaults to CPU count.

    Returns
    -------
    list[dict[str, Any] | None]
        List of feature dictionaries (or None for short pieces)
    """
    if n_workers is None:
        n_workers = cpu_count()

    with Pool(n_workers) as pool:
        results = pool.map(extract_features, pieces)

    return results
