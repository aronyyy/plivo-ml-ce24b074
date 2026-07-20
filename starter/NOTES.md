# NOTES

The model relies on causal prosodic cues extracted from the last 2 seconds of
speech before each pause: energy trajectory across multiple windows (150ms to
800ms) including its slope and drop-into-pause, log-domain pitch level and
slope (falling pitch signals a completed statement, flat/rising suggests
continuation), voicing fraction, final-voiced-run duration (final-syllable
lengthening), and a speech-rate proxy. A Random Forest classifier trained on
these 24 features, pooled across English and Hindi, cut mean response delay
from a ~1600ms silence-only baseline to ~522-599ms at the same 5% false-cutoff
budget. It still fails on pauses where a speaker trails off mid-thought with
naturally falling energy and pitch (e.g. "so I wanted to order... two
pizzas") — these look identical to a real end-of-turn on prosody alone and
would need lexical/semantic content (which this audio-only pipeline
deliberately excludes) to disambiguate. We also tried adding MFCC and log-mel
filterbank statistics, which *hurt* performance (44 features on ~200 turns
overfit/diluted the model) — a useful negative result, reverted. With one
more day: collect more training turns to support the higher-dimensional
spectral features properly, add a lightweight per-language calibration layer
on top of the pooled model, and build a small manually-labeled error set from
the worst-scoring pauses to target feature design more precisely.
