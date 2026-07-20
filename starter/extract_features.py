"""
Causal prosodic feature extraction for end-of-turn detection.

Hard rule: every feature here uses ONLY audio[0 : pause_start].
Nothing after the pause is ever touched.
"""
import numpy as np
from features import speech_before, frame_energy_db, f0_contour, frames

FEATURE_NAMES = [
    "energy_last200ms",       # 0
    "energy_slope",           # 1
    "energy_var_last500ms",   # 2
    "f0_last_voiced_mean",    # 3
    "f0_slope_last400ms",     # 4
    "f0_final_vs_mean",       # 5
    "voiced_frac_last600ms",  # 6
    "final_voiced_run_s",     # 7
    "silence_ratio_last500ms",# 8
    "speech_rate_proxy",      # 9
    "context_dur_s",          # 10
    "pause_position_norm",    # 11
    "energy_drop_ratio",      # 12
    "f0_range_last1s",        # 13
]
N_FEATURES = len(FEATURE_NAMES)


def _safe_slope(y):
    """Least-squares slope of y against its own index. 0 if too short."""
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    x = x - x.mean()
    denom = (x ** 2).sum()
    if denom < 1e-8:
        return 0.0
    return float((x * (y - y.mean())).sum() / denom)


def extract_features(x, sr, pause_start, turn_so_far_s=None):
    """
    x, sr: full audio array (we only ever slice up to pause_start below)
    pause_start: seconds
    turn_so_far_s: optional, total elapsed turn time for position feature
    """
    seg = speech_before(x, sr, pause_start, window_s=2.0)  # <=2s, all causal
    feat = np.zeros(N_FEATURES, dtype=np.float32)

    if len(seg) < sr // 10:  # less than 100ms of context, bail with zeros
        return feat

    # ---- energy features ----
    e_db = frame_energy_db(seg, sr)  # frame hop = 10ms
    if len(e_db) == 0:
        return feat

    frames_per_200ms = max(1, int(0.2 / 0.010))
    frames_per_500ms = max(1, int(0.5 / 0.010))

    feat[0] = e_db[-frames_per_200ms:].mean()
    tail_e = e_db[-frames_per_500ms:] if len(e_db) >= frames_per_500ms else e_db
    feat[1] = _safe_slope(tail_e)
    feat[2] = float(np.var(tail_e))

    # energy drop ratio: last 200ms vs the 200ms window before that
    if len(e_db) >= 2 * frames_per_200ms:
        recent = e_db[-frames_per_200ms:].mean()
        prior = e_db[-2 * frames_per_200ms:-frames_per_200ms].mean()
        feat[12] = float(recent - prior)  # negative => energy falling into pause
    else:
        feat[12] = 0.0

    # ---- pitch features ----
    f0 = f0_contour(seg, sr, frame_ms=40, hop_ms=10)
    voiced_mask = f0 > 0
    voiced_vals = f0[voiced_mask]

    if len(voiced_vals) > 0:
        feat[3] = float(voiced_vals[-3:].mean())
    else:
        feat[3] = 0.0

    frames_per_400ms = max(1, int(0.4 / 0.010))
    tail_f0_region = f0[-frames_per_400ms:] if len(f0) >= frames_per_400ms else f0
    tail_voiced = tail_f0_region[tail_f0_region > 0]
    feat[4] = _safe_slope(tail_voiced) if len(tail_voiced) >= 2 else 0.0

    if len(voiced_vals) >= 4:
        overall_mean = voiced_vals.mean()
        final_mean = voiced_vals[-2:].mean()
        feat[5] = float(final_mean - overall_mean)  # negative => pitch falling (statement-like)
    else:
        feat[5] = 0.0

    frames_per_600ms = max(1, int(0.6 / 0.010))
    tail_voicing_region = voiced_mask[-frames_per_600ms:] if len(voiced_mask) >= frames_per_600ms else voiced_mask
    feat[6] = float(tail_voicing_region.mean()) if len(tail_voicing_region) > 0 else 0.0

    # final voiced run length (final-syllable lengthening signal)
    run = 0
    for v in voiced_mask[::-1]:
        if v:
            run += 1
        else:
            break
    feat[7] = run * 0.010  # seconds

    # silence ratio in last 500ms (energy below a floor relative to segment)
    floor = np.percentile(e_db, 20)
    tail_e2 = e_db[-frames_per_500ms:] if len(e_db) >= frames_per_500ms else e_db
    feat[8] = float((tail_e2 < floor + 3).mean())

    # speaking rate proxy: voiced-frame transitions per second (rough syllable proxy)
    transitions = np.sum(np.abs(np.diff(voiced_mask.astype(np.int8))))
    feat[9] = float(transitions / max(len(seg) / sr, 1e-3))

    feat[10] = float(len(seg) / sr)

    feat[11] = float(turn_so_far_s) if turn_so_far_s is not None else float(pause_start)

    frames_per_1s = max(1, int(1.0 / 0.010))
    tail_f0_1s = f0[-frames_per_1s:] if len(f0) >= frames_per_1s else f0
    voiced_1s = tail_f0_1s[tail_f0_1s > 0]
    feat[13] = float(voiced_1s.max() - voiced_1s.min()) if len(voiced_1s) >= 2 else 0.0

    return feat
