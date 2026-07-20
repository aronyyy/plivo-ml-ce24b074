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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

from features import load_wav
from extract_features import extract_features, N_FEATURES


def make_model(model_type, C=0.5):
    if model_type == "logreg":
        return LogisticRegression(max_iter=3000, class_weight="balanced", C=C)
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=3,
            class_weight="balanced", random_state=0, n_jobs=-1)
    if model_type == "gb":
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05,
            subsample=0.8, random_state=0)
    if model_type == "ensemble":
        logreg = LogisticRegression(max_iter=3000, class_weight="balanced", C=C)
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=4,
            max_features=0.5,  # see more features per split given high dim
            class_weight="balanced", random_state=0, n_jobs=-1)
        gb = GradientBoostingClassifier(
            n_estimators=150, max_depth=2, learning_rate=0.05,
            subsample=0.8, random_state=0)
        return VotingClassifier(
            estimators=[("logreg", logreg), ("rf", rf), ("gb", gb)],
            voting="soft")
    raise ValueError(model_type)


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
    ap.add_argument("--model_type", default="logreg", choices=["logreg", "rf", "gb", "ensemble"])
    ap.add_argument("--top_k", type=int, default=None,
                     help="if set, keep only the top-K most important features "
                          "(selected via a quick RF fit) before final training. "
                          "Recommended when feature count is large relative to "
                          "sample size, e.g. --top_k 15")
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

    selected_idx = None
    if args.top_k is not None:
        from extract_features import FEATURE_NAMES
        probe = RandomForestClassifier(n_estimators=300, max_depth=5,
                                         class_weight="balanced", random_state=0, n_jobs=-1)
        probe.fit(StandardScaler().fit_transform(X), y)
        importances = probe.feature_importances_
        selected_idx = np.argsort(-importances)[:args.top_k]
        print(f"\nselected top {args.top_k} features:")
        for i in selected_idx:
            print(f"  {FEATURE_NAMES[i]:28s} {importances[i]:.3f}")
        X = X[:, selected_idx]
        lo = lo[selected_idx]
        hi = hi[selected_idx]

    # small C sweep via grouped CV (feature quality matters more than this,
    # but it's a cheap win)
    gkf = GroupKFold(n_splits=min(5, len(set(groups))))
    best_C = 0.5
    if args.model_type == "logreg":
        candidate_Cs = [0.1, 0.3, 0.5, 1.0, 2.0]
        best_cv_acc = -1.0
        for C in candidate_Cs:
            accs = []
            for tr_i, te_i in gkf.split(X, y, groups):
                sc = StandardScaler().fit(X[tr_i])
                c = make_model("logreg", C=C)
                c.fit(sc.transform(X[tr_i]), y[tr_i])
                accs.append(c.score(sc.transform(X[te_i]), y[te_i]))
            mean_acc = np.mean(accs)
            print(f"  C={C:<5} 5-fold CV accuracy: mean={mean_acc:.3f} std={np.std(accs):.3f}")
            if mean_acc > best_cv_acc:
                best_cv_acc, best_C = mean_acc, C
        print(f"selected C={best_C} (best CV accuracy={best_cv_acc:.3f})")
    else:
        accs = []
        for tr_i, te_i in gkf.split(X, y, groups):
            sc = StandardScaler().fit(X[tr_i])
            c = make_model(args.model_type)
            c.fit(sc.transform(X[tr_i]), y[tr_i])
            accs.append(c.score(sc.transform(X[te_i]), y[te_i]))
        print(f"{args.model_type} 5-fold CV accuracy: mean={np.mean(accs):.3f} std={np.std(accs):.3f}")

    # held-out check, split by TURN so no leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    tr_idx, te_idx = next(gss.split(X, y, groups))

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[tr_idx])
    X_te = scaler.transform(X[te_idx])

    clf = make_model(args.model_type, C=best_C)
    clf.fit(X_tr, y[tr_idx])
    held_out_acc = clf.score(X_te, y[te_idx])
    print(f"held-out turn accuracy: {held_out_acc:.3f} "
          f"(chance ~ {max(y.mean(), 1 - y.mean()):.3f})")

    # refit on ALL data for the shipped model
    scaler_final = StandardScaler()
    X_all = scaler_final.fit_transform(X)
    clf_final = make_model(args.model_type, C=best_C)
    clf_final.fit(X_all, y)

    with open(args.model_out, "wb") as f:
        pickle.dump({"scaler": scaler_final, "clf": clf_final,
                     "n_features": N_FEATURES,
                     "clip_lo": lo, "clip_hi": hi,
                     "selected_idx": selected_idx}, f)
    print(f"saved model -> {args.model_out}")

    # print feature importance
    from extract_features import FEATURE_NAMES
    if hasattr(clf_final, "coef_"):
        coefs = clf_final.coef_[0]
        order = np.argsort(-np.abs(coefs))
        print("\ntop features by |coef| (standardized):")
        for i in order[:8]:
            print(f"  {FEATURE_NAMES[i]:28s} {coefs[i]:+.3f}")
    elif hasattr(clf_final, "feature_importances_"):
        imp = clf_final.feature_importances_
        order = np.argsort(-imp)
        print("\ntop features by importance:")
        for i in order[:8]:
            print(f"  {FEATURE_NAMES[i]:28s} {imp[i]:.3f}")


if __name__ == "__main__":
    main()