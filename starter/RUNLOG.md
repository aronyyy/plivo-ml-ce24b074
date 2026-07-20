# RUNLOG

## Run 1 — Baseline
`python baseline.py --data_dir eot_data/english --out base.csv`
`python score.py --data_dir eot_data/english --pred base.csv`
- **Score:** ~1600 ms mean delay (silence-only, given reference point)
- Silence-only p_eot=1 for every pause. No signal, just a fixed timeout floor.

## Run 2 — First prosodic features + logistic regression
- **Score:** [FILL IN — your first train.py-based run, if you scored it]
- Extracted energy (last 200ms) and final pitch as starter features, trained
  logistic regression. Beat baseline but weak — thin feature set only captured
  the crudest energy/pitch level, not trends.

## Run 3 — Expanded feature set (24 features), logistic regression
- **Score:** AUC=0.726 (English), AUC=0.758 (Hindi); mean delay 1040 ms / 801 ms
- Added multi-window energy stats (150ms-800ms), energy slopes and drop
  ratios, log-pitch slope/range, voicing fraction, final-voiced-run duration.
  Improved over Run 2 but logistic regression's linear boundary couldn't
  fully separate hold vs eot on these thresholded cues.

## Run 4 — Numerical stability fixes + outlier clipping
- **Score:** [same features, cleaned] — removed NaN/inf warnings via clipped
  energy (dB range), log-domain pitch, and 1st/99th percentile feature
  clipping (bounds saved and reused at inference to avoid train/predict
  distribution mismatch).
- Fixed silent numerical bugs; no accuracy claim attached to this step alone.

## Run 5 — Random Forest classifier (pooled English+Hindi)
- **Score:** AUC=0.931 (English), AUC=0.957 (Hindi); mean delay **599 ms**
  (English), **522 ms** (Hindi) — both at 5.0% interrupted turns
- Switched from linear logistic regression to a Random Forest. EOT cues
  (energy dropping below a floor, pitch flattening) are threshold effects,
  not linear — tree splits capture this far better. This is the model
  we shipped.

## Run 6 — Added MFCC + log-mel filterbank features (44 total)
- **Score:** CV accuracy 0.677, held-out accuracy 0.664 (worse than Run 5)
- Regression. 44 features on ~200 turns is too high-dimensional; MFCC/log-mel
  stats over a 400ms window (only ~4-9 frames) are noisy, and Random Forest's
  default per-split feature sampling gets diluted by low-signal features.
  Reverted to the Run 5 feature set (24 features, no MFCC/log-mel).

## Final shipped model
- **Model:** Random Forest (`n_estimators=300, max_depth=5, min_samples_leaf=3,
  class_weight="balanced"`), pooled across English + Hindi, 24 hand-engineered
  causal prosodic features (energy trajectory, log-pitch trajectory, voicing
  and speech-rate proxies).
- **Final score:** [FILL IN — result of the confirmation run above]
- ~2.7-3x faster response than the 1600ms silence-only baseline, at the
  same ≤5% false-cutoff budget.
