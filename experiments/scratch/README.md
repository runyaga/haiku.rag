# scratch — exploratory probes, NOT part of the harness

These are the ad-hoc scripts that produced the early findings. They are kept for
provenance and are deliberately excluded from `make lint` and `make check`.

**They are not maintained and may contradict `src/fusionlab/`.** A stale copy of
`fusion.py` lived here with a divergent rule — a flat score span normalised to
1.0, where `src/fusionlab/fusion.py` returns 0.0 so a uniformly-bad source is not
promoted. It has been deleted rather than kept, because two implementations of
one idea is how a corrected bug comes back.

Anything worth keeping belongs in `src/fusionlab/` with a test.
