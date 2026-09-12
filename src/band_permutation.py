"""Exact permutation test on the session-level band table.

Nine sessions, four queenright and five queenless, is the sample size behind
every class comparison in results/band_table.json.  With n that small the
question "is the difference between the classes larger than chance" has an
exact answer: there are only C(9, 4) = 126 ways to hand out the four
queenright labels, so every relabelling can be enumerated and the observed
gap placed among them.  No distribution is assumed.

For each band the script reports the observed difference of session means
(queenright minus queenless), the two-sided exact p-value over the 126
relabellings, the session-level AUC (the share of the 4 x 5 = 20
queenright/queenless session pairs in which the queenright session has the
larger share), and the same numbers with the 6-clip session CF001 dropped
(8 sessions, C(8, 4) = 70 relabellings), because a session that contributes
6 clips carries the same weight as one that contributes 1200 in every
session-level mean on the page.

Reads results/band_table.json, writes results/band_permutation.json.
Standard library only.
"""
import json
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = {"session", "hive", "source", "queen", "n_clips"}


def mean(xs):
    return sum(xs) / len(xs)


def exact_test(values, labels):
    """values, labels: lists over sessions; labels are 1 (queenright) / 0."""
    n = len(values)
    k = sum(labels)
    obs = mean([v for v, y in zip(values, labels) if y == 1]) - mean(
        [v for v, y in zip(values, labels) if y == 0]
    )
    gaps = []
    for pos in combinations(range(n), k):
        pos = set(pos)
        a = [values[i] for i in range(n) if i in pos]
        b = [values[i] for i in range(n) if i not in pos]
        gaps.append(mean(a) - mean(b))
    p_two = sum(abs(g) >= abs(obs) - 1e-12 for g in gaps) / len(gaps)
    pairs = [(v1, v0) for v1, y1 in zip(values, labels) if y1 == 1
             for v0, y0 in zip(values, labels) if y0 == 0]
    auc = mean([1.0 if v1 > v0 else 0.5 if v1 == v0 else 0.0 for v1, v0 in pairs])
    return {
        "queenright_mean": mean([v for v, y in zip(values, labels) if y == 1]),
        "queenless_mean": mean([v for v, y in zip(values, labels) if y == 0]),
        "difference": obs,
        "exact_p_two_sided": p_two,
        "relabellings": len(gaps),
        "session_auc": auc,
        "n_sessions": n,
    }


def main():
    rows = json.loads((ROOT / "results" / "band_table.json").read_text())
    bands = [k for k in rows[0] if k not in META]
    out = {"bands": {}, "note": __doc__.strip().splitlines()[0]}
    for band in bands:
        full = exact_test([r[band] for r in rows], [r["queen"] for r in rows])
        kept = [r for r in rows if r["n_clips"] >= 100]
        sub = exact_test([r[band] for r in kept], [r["queen"] for r in kept])
        out["bands"][band] = {"all_sessions": full, "sessions_with_100_plus_clips": sub}
        print(f"{band:42s} diff {full['difference']:+.3f}  p {full['exact_p_two_sided']:.3f}"
              f"  AUC {full['session_auc']:.2f}   | without CF001: p {sub['exact_p_two_sided']:.3f}"
              f"  AUC {sub['session_auc']:.2f}")
    out["dropped_sessions"] = [r["session"] for r in rows if r["n_clips"] < 100]
    out["smallest_possible_p"] = 1 / 126   # 4 vs 5 is unbalanced: only the observed labelling reaches its own gap
    (ROOT / "results" / "band_permutation.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
