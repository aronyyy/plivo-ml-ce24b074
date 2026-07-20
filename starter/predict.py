"""
Required deliverable. Loads a SAVED model (trained by train_model.py) and
scores an unseen data_dir with the same folder/labels schema.

Usage:
    python predict.py --data_dir eot_data/english --out predictions.csv
    python predict.py --data_dir eot_data/hindi   --out predictions_hi.csv

By default loads model_pooled.pkl. Pass --model to use a different one.
"""
import argparse
import csv
import os
import pickle

import numpy as np

from features import load_wav
from extract_features import extract_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--model", default="model_pooled.pkl")
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        bundle = pickle.load(f)
    scaler, clf = bundle["scaler"], bundle["clf"]
    clip_lo = bundle.get("clip_lo")
    clip_hi = bundle.get("clip_hi")
    selected_idx = bundle.get("selected_idx")

    labels_path = os.path.join(args.data_dir, "labels.csv")
    rows = list(csv.DictReader(open(labels_path)))

    by_turn = {}
    for r in rows:
        by_turn.setdefault(r["turn_id"], []).append(r)

    cache = {}
    out_rows = []
    for turn_id, turn_rows in by_turn.items():
        turn_rows = sorted(turn_rows, key=lambda r: int(r["pause_index"]))
        path = os.path.join(args.data_dir, turn_rows[0]["audio_file"])
        if path not in cache:
            cache[path] = load_wav(path)
        x, sr = cache[path]

        feats = []
        for r in turn_rows:
            pause_start = float(r["pause_start"])
            feat = extract_features(x, sr, pause_start, turn_so_far_s=pause_start)
            feats.append(feat)

        feats = np.array(feats, dtype=np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        if selected_idx is not None:
            feats = feats[:, selected_idx]
        if clip_lo is not None and clip_hi is not None:
            feats = np.clip(feats, clip_lo, clip_hi)
        feats_scaled = scaler.transform(feats)
        probs = clf.predict_proba(feats_scaled)[:, 1]

        for r, p in zip(turn_rows, probs):
            out_rows.append({
                "turn_id": r["turn_id"],
                "pause_index": r["pause_index"],
                "p_eot": f"{p:.4f}",
            })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["turn_id", "pause_index", "p_eot"])
        w.writeheader()
        w.writerows(out_rows)

    print(f"wrote {len(out_rows)} predictions -> {args.out}")


if __name__ == "__main__":
    main()