"""
Train a logistic regression classifier on prosodic features and save it.

Usage:
    python train_model.py --data_dir eot_data/english --model_out model_en.pkl
    python train_model.py --data_dir eot_data/hindi   --model_out model_hi.pkl

Or train one pooled model on both (often better given small per-language data):
    python train_model.py --data_dir eot_data/english eot_data/hindi --model_out model_pooled.pkl
"""
import argparse
import csv
import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

from features import load_wav
from extract_features import extract_features, N_FEATURES


def build_dataset(data_dirs):
    X, y, groups, keys, meta = [], [], [], [], []
    for data_dir in data_dirs:
        labels_path = os.path.join(data_dir, "labels.csv")
        rows = list(csv.DictReader(open(labels_path)))
        cache = {}
        # group rows by turn to compute "turn_so_far" context cheaply
        by_turn = {}
        for r in rows:
            by_turn.setdefault(r["turn_id"], []).append(r)

        for turn_id, turn_rows in by_turn.items():
            turn_rows = sorted(turn_rows, key=lambda r: int(r["pause_index"]))
            path = os.path.join(data_dir, turn_rows[0]["audio_file"])
            if path not in cache:
                cache[path] = load_wav(path)
            x, sr = cache[path]

            for r in turn_rows:
                pause_start = float(r["pause_start"])
                feat = extract_features(x, sr, pause_start, turn_so_far_s=pause_start)
                X.append(feat)
                y.append(1 if r["label"] == "eot" else 0)
                # groups must be unique across pooled languages
                groups.append(f"{os.path.basename(data_dir)}::{turn_id}")
                keys.append((turn_id, int(r["pause_index"])))
                meta.append(data_dir)

    return np.array(X, dtype=np.float32), np.array(y), groups, keys, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, nargs="+",
                     help="one or more data dirs, e.g. eot_data/english eot_data/hindi")
    ap.add_argument("--model_out", required=True)
    args = ap.parse_args()

    X, y, groups, keys, meta = build_dataset(args.data_dir)
    print(f"dataset: {X.shape[0]} pauses, {len(set(groups))} turns, "
          f"pos_rate={y.mean():.3f}")

    # numerical safety net: kill any leftover NaN/inf, clip extreme outliers
    # per-feature to the 1st/99th percentile (guards against divide-by-zero
    # or empty-window artifacts leaking into a few rows)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    lo = np.percentile(X, 1, axis=0)
    hi = np.percentile(X, 99, axis=0)
    X = np.clip(X, lo, hi)

    # small C sweep via grouped CV (feature quality matters more than this,
    # but it's a cheap win)
    gkf = GroupKFold(n_splits=min(5, len(set(groups))))
    candidate_Cs = [0.1, 0.3, 0.5, 1.0, 2.0]
    best_C, best_cv_acc = 0.5, -1.0
    for C in candidate_Cs:
        accs = []
        for tr_i, te_i in gkf.split(X, y, groups):
            sc = StandardScaler().fit(X[tr_i])
            c = LogisticRegression(max_iter=3000, class_weight="balanced", C=C)
            c.fit(sc.transform(X[tr_i]), y[tr_i])
            accs.append(c.score(sc.transform(X[te_i]), y[te_i]))
        mean_acc = np.mean(accs)
        print(f"  C={C:<5} 5-fold CV accuracy: mean={mean_acc:.3f} std={np.std(accs):.3f}")
        if mean_acc > best_cv_acc:
            best_cv_acc, best_C = mean_acc, C
    print(f"selected C={best_C} (best CV accuracy={best_cv_acc:.3f})")

    # held-out check, split by TURN so no leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    tr_idx, te_idx = next(gss.split(X, y, groups))

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[tr_idx])
    X_te = scaler.transform(X[te_idx])

    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=best_C)
    clf.fit(X_tr, y[tr_idx])
    held_out_acc = clf.score(X_te, y[te_idx])
    print(f"held-out turn accuracy: {held_out_acc:.3f} "
          f"(chance ~ {max(y.mean(), 1 - y.mean()):.3f})")

    # refit on ALL data for the shipped model
    scaler_final = StandardScaler()
    X_all = scaler_final.fit_transform(X)
    clf_final = LogisticRegression(max_iter=3000, class_weight="balanced", C=best_C)
    clf_final.fit(X_all, y)

    with open(args.model_out, "wb") as f:
        pickle.dump({"scaler": scaler_final, "clf": clf_final,
                     "n_features": N_FEATURES,
                     "clip_lo": lo, "clip_hi": hi}, f)
    print(f"saved model -> {args.model_out}")

    # print feature importance (coef magnitude, since features are standardized)
    from extract_features import FEATURE_NAMES
    coefs = clf_final.coef_[0]
    order = np.argsort(-np.abs(coefs))
    print("\ntop features by |coef| (standardized):")
    for i in order[:8]:
        print(f"  {FEATURE_NAMES[i]:28s} {coefs[i]:+.3f}")


if __name__ == "__main__":
    main()