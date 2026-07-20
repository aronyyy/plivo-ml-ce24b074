"""
Causal prosodic feature extraction for end-of-turn detection.

Hard rule: every feature here uses ONLY audio[0 : pause_start].
Nothing after the pause is ever touched.

v2: expanded energy family (top feature was energy_last200ms), multiple
window sizes, and numerical-stability fixes (no NaN/inf, clipped ranges,
log-domain pitch).
"""
import numpy as np
from features import speech_before, frame_energy_db, f0_contour, frames

FEATURE_NAMES = [
    "energy_last150ms",        # 0
    "energy_last200ms",        # 1
    "energy_last300ms",        # 2
    "energy_last500ms",        # 3
    "energy_last800ms",        # 4
    "energy_slope_300ms",      # 5
    "energy_slope_500ms",      # 6
    "energy_slope_800ms",      # 7
    "energy_var_last500ms",    # 8
    "energy_min_last300ms",    # 9
    "energy_max_minus_last",   # 10
    "energy_drop_ratio_200",   # 11
    "energy_drop_ratio_400",   # 12
    "energy_rel_to_turn_mean", # 13
    "f0_last_voiced_mean",     # 14 (log-Hz, semitone-ish)
    "f0_slope_last400ms",      # 15
    "f0_final_vs_mean",        # 16
    "f0_range_last1s",         # 17
    "voiced_frac_last600ms",   # 18
    "final_voiced_run_s",      # 19
    "silence_ratio_last500ms", # 20
    "speech_rate_proxy",       # 21
    "context_dur_s",           # 22
    "pause_position_norm",     # 23
]
N_FEATURES = len(FEATURE_NAMES)

E_DB_MIN, E_DB_MAX = -80.0, 0.0
F0_MIN_HZ, F0_MAX_HZ = 60.0, 400.0


def _safe_slope(y):
    y = np.asarray(y, dtype=np.float32)
    y = y[np.isfinite(y)]
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=np.float32)
    x = x - x.mean()
    denom = (x ** 2).sum()
    if denom < 1e-8:
        return 0.0
    slope = float((x * (y - y.mean())).sum() / denom)
    return slope if np.isfinite(slope) else 0.0


def _safe_mean(y, default=0.0):
    y = np.asarray(y, dtype=np.float32)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return default
    return float(y.mean())


def _clip_e(e):
    return np.clip(e, E_DB_MIN, E_DB_MAX)


def _log_f0(f0_hz):
    out = np.zeros_like(f0_hz, dtype=np.float32)
    voiced = f0_hz > 0
    vals = np.clip(f0_hz[voiced], F0_MIN_HZ, F0_MAX_HZ)
    out[voiced] = np.log2(vals)
    return out


def extract_features(x, sr, pause_start, turn_so_far_s=None):
    seg = speech_before(x, sr, pause_start, window_s=2.0)
    feat = np.zeros(N_FEATURES, dtype=np.float32)

    if len(seg) < sr // 10:
        return feat

    hop_s = 0.010

    e_db_raw = frame_energy_db(seg, sr)
    if len(e_db_raw) == 0:
        return feat
    e_db = _clip_e(e_db_raw)

    def tail(ms):
        n = max(1, int((ms / 1000.0) / hop_s))
        return e_db[-n:] if len(e_db) >= n else e_db

    feat[0] = _safe_mean(tail(150))
    feat[1] = _safe_mean(tail(200))
    feat[2] = _safe_mean(tail(300))
    feat[3] = _safe_mean(tail(500))
    feat[4] = _safe_mean(tail(800))

    feat[5] = _safe_slope(tail(300))
    feat[6] = _safe_slope(tail(500))
    feat[7] = _safe_slope(tail(800))

    var500 = tail(500)
    feat[8] = float(np.var(var500)) if len(var500) > 1 else 0.0

    min300 = tail(300)
    feat[9] = float(np.min(min300)) if len(min300) > 0 else 0.0

    feat[10] = float(np.max(e_db) - feat[1])

    def drop_ratio(ms):
        n = max(1, int((ms / 1000.0) / hop_s))
        if len(e_db) >= 2 * n:
            recent = e_db[-n:].mean()
            prior = e_db[-2 * n:-n].mean()
            return float(recent - prior)
        return 0.0

    feat[11] = drop_ratio(200)
    feat[12] = drop_ratio(400)

    turn_mean_e = _safe_mean(e_db)
    feat[13] = feat[1] - turn_mean_e

    f0_raw = f0_contour(seg, sr, frame_ms=40, hop_ms=10)
    f0_log = _log_f0(f0_raw)
    voiced_mask = f0_raw > 0
    voiced_log = f0_log[voiced_mask]

    feat[14] = _safe_mean(voiced_log[-3:]) if len(voiced_log) >= 1 else 0.0

    n400 = max(1, int(0.4 / hop_s))
    tail_f0_log = f0_log[-n400:] if len(f0_log) >= n400 else f0_log
    tail_voiced_log = tail_f0_log[tail_f0_log > 0]
    feat[15] = _safe_slope(tail_voiced_log) if len(tail_voiced_log) >= 2 else 0.0

    if len(voiced_log) >= 4:
        feat[16] = _safe_mean(voiced_log[-2:]) - _safe_mean(voiced_log)
    else:
        feat[16] = 0.0

    n1s = max(1, int(1.0 / hop_s))
    tail_f0_1s = f0_log[-n1s:] if len(f0_log) >= n1s else f0_log
    voiced_1s = tail_f0_1s[tail_f0_1s > 0]
    feat[17] = float(voiced_1s.max() - voiced_1s.min()) if len(voiced_1s) >= 2 else 0.0

    n600 = max(1, int(0.6 / hop_s))
    tail_v600 = voiced_mask[-n600:] if len(voiced_mask) >= n600 else voiced_mask
    feat[18] = float(tail_v600.mean()) if len(tail_v600) > 0 else 0.0

    run = 0
    for v in voiced_mask[::-1]:
        if v:
            run += 1
        else:
            break
    feat[19] = run * hop_s

    floor = np.percentile(e_db, 20)
    n500 = max(1, int(0.5 / hop_s))
    tail_e500 = e_db[-n500:] if len(e_db) >= n500 else e_db
    feat[20] = float((tail_e500 < floor + 3).mean())

    transitions = np.sum(np.abs(np.diff(voiced_mask.astype(np.int8))))
    feat[21] = float(transitions / max(len(seg) / sr, 1e-3))

    feat[22] = float(len(seg) / sr)
    feat[23] = float(turn_so_far_s) if turn_so_far_s is not None else float(pause_start)

    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return feat