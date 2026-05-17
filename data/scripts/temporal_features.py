#!/usr/bin/env python3
"""
Temporal Feature Extraction and Benchmark

Computes features on quartile segments of each piece to capture
temporal evolution (how complexity, density, etc. change over time).
This tests whether the "narrative arc" of a piece carries popularity signal.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from music21 import converter, note, chord
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Features we can compute on segments
SEGMENT_FEATURES = [
    'note_count',
    'pitch_mean',
    'pitch_std',
    'pitch_range',
    'note_density',  # notes per unit time
    'interval_mean',
    'stepwise_ratio',
    'unique_pitches',
]


def compute_segment_features(notes_with_times: list[tuple]) -> dict:
    """
    Compute features for a segment of notes.
    notes_with_times: list of (offset, pitch, duration) tuples
    """
    if len(notes_with_times) < 3:
        return None
    
    offsets = [n[0] for n in notes_with_times]
    pitches = [n[1] for n in notes_with_times]
    
    start_time = min(offsets)
    end_time = max(offsets) + max([n[2] for n in notes_with_times])
    duration = end_time - start_time
    if duration <= 0:
        duration = 1
    
    # Pitch features
    pitch_mean = np.mean(pitches)
    pitch_std = np.std(pitches) if len(pitches) > 1 else 0
    pitch_range = max(pitches) - min(pitches) if pitches else 0
    
    # Interval features
    intervals = [abs(pitches[i] - pitches[i-1]) for i in range(1, len(pitches))]
    interval_mean = np.mean(intervals) if intervals else 0
    stepwise = sum(1 for iv in intervals if iv <= 2)
    stepwise_ratio = stepwise / len(intervals) if intervals else 0
    
    return {
        'note_count': len(notes_with_times),
        'pitch_mean': pitch_mean,
        'pitch_std': pitch_std,
        'pitch_range': pitch_range,
        'note_density': len(notes_with_times) / duration,
        'interval_mean': interval_mean,
        'stepwise_ratio': stepwise_ratio,
        'unique_pitches': len(set(pitches)),
    }


def extract_temporal_features(midi_path: str, n_segments: int = 4) -> dict | None:
    """
    Extract features for each temporal segment of a piece.
    Returns dict with keys like 'pitch_mean_q1', 'pitch_mean_q2', etc.
    Also computes slopes for each feature.
    """
    try:
        score = converter.parse(midi_path)
    except Exception as e:
        return None
    
    # Flatten all notes
    notes_data = []
    for element in score.flatten().notes:
        if isinstance(element, note.Note):
            notes_data.append((float(element.offset), element.pitch.midi, float(element.quarterLength)))
        elif isinstance(element, chord.Chord):
            for p in element.pitches:
                notes_data.append((float(element.offset), p.midi, float(element.quarterLength)))
    
    if len(notes_data) < 20:
        return None
    
    # Sort by offset
    notes_data.sort(key=lambda x: x[0])
    
    # Calculate segment boundaries
    total_duration = notes_data[-1][0] - notes_data[0][0]
    if total_duration <= 0:
        return None
    
    start_offset = notes_data[0][0]
    segment_duration = total_duration / n_segments
    
    # Assign notes to segments
    segments = defaultdict(list)
    for offset, pitch, dur in notes_data:
        seg_idx = min(int((offset - start_offset) / segment_duration), n_segments - 1)
        segments[seg_idx].append((offset, pitch, dur))
    
    # Compute features for each segment
    segment_features = {}
    feature_values = defaultdict(list)  # For computing slopes
    
    for seg_idx in range(n_segments):
        seg_notes = segments.get(seg_idx, [])
        feats = compute_segment_features(seg_notes)
        if feats is None:
            feats = {k: np.nan for k in SEGMENT_FEATURES}
        
        suffix = f'_q{seg_idx + 1}'
        for k, v in feats.items():
            segment_features[k + suffix] = v
            feature_values[k].append(v)
    
    # Compute slopes (linear regression coefficient across segments)
    x = np.arange(n_segments)
    for feat_name, values in feature_values.items():
        values = np.array(values)
        # Handle NaN
        valid = ~np.isnan(values)
        if valid.sum() >= 2:
            slope = np.polyfit(x[valid], values[valid], 1)[0]
        else:
            slope = 0
        segment_features[feat_name + '_slope'] = slope
    
    return segment_features


def main():
    # Load existing dataset for piece IDs and popularity
    df = pd.read_csv('data/features/balanced_corpus_v2.csv')
    
    # Try to load cached temporal features first
    temporal_cache = Path('data/features/temporal_features.csv')
    
    if temporal_cache.exists():
        print("Loading cached temporal features...")
        df_temporal = pd.read_csv(temporal_cache)
        print(f"Loaded {len(df_temporal)} pieces from cache")
    else:
        # Find MIDI files
        midi_dir = Path('data/corpus/midi')
        
        # Build mapping from piece_id to MIDI path
        piece_to_midi = {}
        for midi_path in midi_dir.rglob('*.mid'):
            piece_id = midi_path.stem
            piece_to_midi[piece_id] = str(midi_path)
        
        print(f"Found {len(piece_to_midi)} MIDI files")
        print(f"Dataset has {len(df)} pieces")
        
        # Extract temporal features for each piece
        temporal_data = []
        matched = 0
        
        for idx, row in df.iterrows():
            piece_id = row['piece_id']
            
            # Try to find matching MIDI
            midi_path = piece_to_midi.get(piece_id)
            if midi_path is None:
                # Try alternate matching
                for candidate, path in piece_to_midi.items():
                    if piece_id in candidate or candidate in piece_id:
                        midi_path = path
                        break
            
            if midi_path and Path(midi_path).exists():
                feats = extract_temporal_features(midi_path)
                if feats:
                    feats['piece_id'] = piece_id
                    feats['composer'] = row['composer']
                    feats['popularity'] = row['popularity']
                    temporal_data.append(feats)
                    matched += 1
                    if matched % 50 == 0:
                        print(f"  Processed {matched} pieces...")
        
        print(f"\nMatched {matched} pieces with temporal features")
        
        if matched < 100:
            print("Not enough matched pieces. Trying direct MIDI parsing...")
            temporal_data = []
            matched = 0
            
            for midi_path in midi_dir.rglob('*.mid'):
                feats = extract_temporal_features(str(midi_path))
                if feats:
                    composer = midi_path.parent.name
                    piece_name = midi_path.stem
                    
                    matching = df[df['piece_id'].str.contains(piece_name[:20], na=False)]
                    if len(matching) > 0:
                        row = matching.iloc[0]
                        feats['piece_id'] = row['piece_id']
                        feats['composer'] = row['composer']
                        feats['popularity'] = row['popularity']
                        temporal_data.append(feats)
                        matched += 1
            
            print(f"Matched {matched} pieces with fallback method")
        
        if matched < 50:
            print("\nERROR: Not enough MIDI files could be matched.")
            return
        
        df_temporal = pd.DataFrame(temporal_data)
        
        # Save to cache
        df_temporal.to_csv(temporal_cache, index=False)
        print(f"Saved temporal features to {temporal_cache}")
    
    print(f"\nTemporal features shape: {df_temporal.shape}")
    
    # =============================================
    # BENCHMARK 1: Temporal features alone
    # =============================================
    temporal_cols = [c for c in df_temporal.columns if c not in ['piece_id', 'composer', 'popularity']]
    slope_cols = [c for c in temporal_cols if '_slope' in c]
    print(f"Temporal feature count: {len(temporal_cols)} (including {len(slope_cols)} slopes)")
    
    X_temporal = df_temporal[temporal_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    y = df_temporal['popularity'].values
    composers = df_temporal['composer'].astype('category').cat.codes.values
    
    X_temporal_resid = X_temporal.copy()
    for c in np.unique(composers):
        mask = composers == c
        X_temporal_resid[mask] -= X_temporal[mask].mean(axis=0)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
    
    print("\n" + "=" * 70)
    print("BENCHMARK 1: TEMPORAL FEATURES ALONE")
    print("=" * 70)
    
    s_rf = cross_val_score(rf, X_temporal_resid, y, cv=kf, scoring='r2')
    s_gb = cross_val_score(gb, X_temporal_resid, y, cv=kf, scoring='r2')
    print(f"RF (residualized):  {s_rf.mean()*100:.1f}%")
    print(f"GBR (residualized): {s_gb.mean()*100:.1f}%")
    
    # =============================================
    # BENCHMARK 2: Slopes only (8 features)
    # =============================================
    print("\n" + "=" * 70)
    print("BENCHMARK 2: SLOPES ONLY (8 features)")
    print("=" * 70)
    
    X_slopes = df_temporal[slope_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    X_slopes_resid = X_slopes.copy()
    for c in np.unique(composers):
        mask = composers == c
        X_slopes_resid[mask] -= X_slopes[mask].mean(axis=0)
    
    s_rf_slopes = cross_val_score(rf, X_slopes_resid, y, cv=kf, scoring='r2')
    s_gb_slopes = cross_val_score(gb, X_slopes_resid, y, cv=kf, scoring='r2')
    print(f"RF (residualized):  {s_rf_slopes.mean()*100:.1f}%")
    print(f"GBR (residualized): {s_gb_slopes.mean()*100:.1f}%")
    
    # =============================================
    # BENCHMARK 3: Static + Temporal COMBINED
    # =============================================
    print("\n" + "=" * 70)
    print("BENCHMARK 3: STATIC + TEMPORAL COMBINED")
    print("=" * 70)
    
    # Load static features and merge
    STATIC_FEATURES = [
        'pitch_range', 'pitch_mean', 'pitch_std', 'pitch_class_entropy',
        'interval_mean', 'stepwise_ratio', 'leap_ratio', 'contour_change_rate',
        'total_notes', 'duration', 'note_density', 'unique_durations',
        'duration_std', 'repetition_ratio',
        'chord_change_rate', 'chord_change_std', 'avg_chord_size',
        'avg_vertical_range', 'chord_pc_entropy', 'tonic_dominant_ratio',
        'cadence_rate', 'key_change_rate', 'modulation_mean',
        'sectional_repetition_ratio', 'sectional_similarity_mean'
    ]
    
    # Merge static and temporal on piece_id
    df_merged = df_temporal.merge(df[['piece_id'] + STATIC_FEATURES], on='piece_id', how='inner')
    print(f"Merged dataset size: {len(df_merged)}")
    
    X_static = df_merged[STATIC_FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0).values
    X_temporal_m = df_merged[temporal_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    X_slopes_m = df_merged[slope_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    y_merged = df_merged['popularity'].values
    composers_m = df_merged['composer'].astype('category').cat.codes.values
    
    # Residualize all
    def residualize(X, composers):
        X_r = X.copy()
        for c in np.unique(composers):
            mask = composers == c
            X_r[mask] -= X[mask].mean(axis=0)
        return X_r
    
    X_static_resid = residualize(X_static, composers_m)
    X_temporal_resid_m = residualize(X_temporal_m, composers_m)
    X_slopes_resid_m = residualize(X_slopes_m, composers_m)
    
    # Combined: static + all temporal
    X_combined_all = np.hstack([X_static_resid, X_temporal_resid_m])
    print(f"Static + all temporal: {X_combined_all.shape[1]} features")
    
    s_rf_comb = cross_val_score(rf, X_combined_all, y_merged, cv=kf, scoring='r2')
    s_gb_comb = cross_val_score(gb, X_combined_all, y_merged, cv=kf, scoring='r2')
    print(f"  RF:  {s_rf_comb.mean()*100:.1f}%")
    print(f"  GBR: {s_gb_comb.mean()*100:.1f}%")
    
    # Combined: static + slopes only (33 features - the lean version)
    X_combined_lean = np.hstack([X_static_resid, X_slopes_resid_m])
    print(f"\nStatic + slopes only: {X_combined_lean.shape[1]} features")
    
    s_rf_lean = cross_val_score(rf, X_combined_lean, y_merged, cv=kf, scoring='r2')
    s_gb_lean = cross_val_score(gb, X_combined_lean, y_merged, cv=kf, scoring='r2')
    print(f"  RF:  {s_rf_lean.mean()*100:.1f}%")
    print(f"  GBR: {s_gb_lean.mean()*100:.1f}%")
    
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print("Static only (25 features):          RF=20.4%, GBR=24.6%")
    print(f"Temporal only ({len(temporal_cols)} features):        RF={s_rf.mean()*100:.1f}%, GBR={s_gb.mean()*100:.1f}%")
    print(f"Slopes only (8 features):           RF={s_rf_slopes.mean()*100:.1f}%, GBR={s_gb_slopes.mean()*100:.1f}%")
    print(f"Static + temporal ({X_combined_all.shape[1]} features):   RF={s_rf_comb.mean()*100:.1f}%, GBR={s_gb_comb.mean()*100:.1f}%")
    print(f"Static + slopes ({X_combined_lean.shape[1]} features):     RF={s_rf_lean.mean()*100:.1f}%, GBR={s_gb_lean.mean()*100:.1f}%")
    print("=" * 70)


if __name__ == '__main__':
    main()
