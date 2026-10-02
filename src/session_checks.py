#!/usr/bin/env python3
"""Sections 4 and 5 of the README read session by session.

The page reports one pooled AUC per protocol and describes the wing-beat
distributions in a sentence. This script opens both up, one recording
session at a time, from files that already exist:

  results/cv/session__*.json, results/cv/hive__*.json
      out-of-fold probabilities of the log-mel network (written by train.py)
  results/wingbeat_pinn.json
      per-session wing-beat distributions p(f0) and named quantities
  results/session_meta.json

and writes results/session_checks.json. No audio is needed. numpy only.

    python src/session_checks.py
"""
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def auc(y, s):
    """Mann-Whitney AUC, ties counted half."""
    y = np.asarray(y)
    s = np.asarray(s, float)
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def pair_auc(x, m):
    """Share of (queenright, queenless) session pairs with the larger value
    on the queenright side."""
    a, b = x[m], x[~m]
    return float(np.mean([(u > v) + 0.5 * (u == v) for u in a for v in b]))


def exact_p(x, q, stat):
    """Two-sided exact permutation p over every relabelling of the sessions."""
    n, k = len(x), int(q.sum())
    obs = abs(stat(x, q.astype(bool)))
    hits = tot = 0
    for idx in itertools.combinations(range(n), k):
        m = np.zeros(n, bool)
        m[list(idx)] = True
        hits += abs(stat(x, m)) >= obs - 1e-12
        tot += 1
    return hits / tot, tot


def mean_diff(x, m):
    return x[m].mean() - x[~m].mean()


def cnn_by_session(meta):
    cv = {s: json.loads((RESULTS / "cv" / f"session__{s}.json").read_text())
          for s in meta}
    n = {s: len(cv[s]["y"]) for s in meta}
    N = sum(n.values())
    N1 = sum(n[s] for s in meta if meta[s]["queen"] == 1)
    rows, Y, P, PRIOR = {}, [], [], []
    for s in meta:
        q = meta[s]["queen"]
        p = np.asarray(cv[s]["p"])
        assert cv[s]["n_train"] == N - n[s]
        prior = (N1 - n[s] * q) / (N - n[s])
        called = float((p > 0.5).mean())
        rows[s] = {
            "queen": q, "hive": meta[s]["hive"], "source": meta[s]["source"],
            "n": n[s], "n_train": cv[s]["n_train"],
            "train_prior_queenright": prior,
            "called_queenright": called,
            "n_called_queenright": int((p > 0.5).sum()),
            "median_p": float(np.median(p)), "mean_p": float(p.mean()),
            "majority_correct": bool((called > 0.5) == (q == 1)),
        }
        Y += list(cv[s]["y"]); P += list(p); PRIOR += [prior] * n[s]
    names = list(meta)
    q = np.array([meta[s]["queen"] for s in names])
    mp = np.array([rows[s]["mean_p"] for s in names])
    p_exact, tot = exact_p(mp, q, lambda x, m: pair_auc(x, m) - 0.5)
    summary = {
        "n_clips": N, "n_queenright_clips": N1,
        "pooled_auc": auc(Y, P),
        "sessions_majority_correct": int(sum(r["majority_correct"] for r in rows.values())),
        "sessions_called_wrong_over_80pct": int(sum(
            (r["called_queenright"] > 0.8 and r["queen"] == 0)
            or (r["called_queenright"] < 0.2 and r["queen"] == 1) for r in rows.values())),
        "session_auc_of_mean_p": pair_auc(mp, q.astype(bool)),
        "session_auc_exact_p_two_sided": p_exact, "relabellings": tot,
        "prior_only_pooled_auc": auc(Y, PRIOR),
        "train_prior_when_queenright_held_out": sorted(
            rows[s]["train_prior_queenright"] for s in names if meta[s]["queen"] == 1),
        "train_prior_when_queenless_held_out": sorted(
            rows[s]["train_prior_queenright"] for s in names if meta[s]["queen"] == 0),
    }
    return rows, summary


def cnn_by_hive(meta):
    out = {}
    for h in sorted({m["hive"] for m in meta.values()}):
        d = json.loads((RESULTS / "cv" / f"hive__{h}.json").read_text())
        y, p = np.asarray(d["y"]), np.asarray(d["p"])
        row = {"n_train": d["n_train"], "classes": {}}
        for c in (0, 1):
            if (y == c).any():
                row["classes"][str(c)] = {
                    "n": int((y == c).sum()),
                    "called_queenright": float((p[y == c] > 0.5).mean()),
                    "median_p": float(np.median(p[y == c])),
                }
        row["called_queenright_all"] = float((p > 0.5).mean())
        if len(row["classes"]) == 2:
            row["within_hive_auc"] = auc(y, p)
        out[h] = row
    return out


def wingbeat_by_session(wb):
    g = np.asarray(wb["f0_grid"])
    rows = {}
    for s, v in wb["per_session"].items():
        p = np.asarray(v["p_f0"], float)
        p = p / p.sum()
        mass = lambda lo, hi: float(p[(g >= lo) & (g <= hi)].sum())
        at = lambda f: float(p[g == f][0])
        rows[s] = {
            "queen": v["queen"], "n": v["n"],
            "f0_mean": v["f0_mean"], "comb_share": v["comb_share"],
            "mass_80_95": mass(80, 95), "mass_100_160": mass(100, 160),
            "mass_180_260": mass(180, 260),
            "p_at_80": at(80), "p_at_100": at(100), "p_at_150": at(150),
            "line_ratio_100": at(100) / (0.5 * (at(95) + at(105))),
            "top_bin_hz": float(g[int(np.argmax(p))]), "top_bin_p": float(p.max()),
        }
    return rows


def named_quantities(wb):
    names = list(wb["per_session"])
    q = np.array([wb["per_session"][s]["queen"] for s in names])
    keep = np.array([s != "CF001" for s in names])
    out = {}
    for k in ("f0_mean", "f0_sd", "comb_share", "bg_slope"):
        x = np.array([wb["per_session"][s][k] for s in names], float)
        p_all, tot = exact_p(x, q, mean_diff)
        p_wo, tot_wo = exact_p(x[keep], q[keep], mean_diff)
        out[k] = {
            "queenright_mean": float(x[q == 1].mean()),
            "queenless_mean": float(x[q == 0].mean()),
            "queenless_mean_without_CF001": float(x[keep][q[keep] == 0].mean()),
            "session_auc": pair_auc(x, q.astype(bool)),
            "exact_p": p_all, "relabellings": tot,
            "exact_p_without_CF001": p_wo, "relabellings_without_CF001": tot_wo,
        }
    return out


def main():
    meta = json.loads((RESULTS / "session_meta.json").read_text())
    wb = json.loads((RESULTS / "wingbeat_pinn.json").read_text())
    rows, summary = cnn_by_session(meta)
    out = {
        "cnn_session_protocol": {"per_session": rows, "summary": summary},
        "cnn_hive_protocol": cnn_by_hive(meta),
        "wingbeat_per_session": wingbeat_by_session(wb),
        "named_quantities_session_level": named_quantities(wb),
        "n_clips": {"cnn": summary["n_clips"], "wingbeat_and_band_rows": wb["n_clips"]},
    }
    (RESULTS / "session_checks.json").write_text(json.dumps(out, indent=1))
    s = summary
    print(f"CNN, session protocol: pooled AUC {s['pooled_auc']:.3f}, "
          f"{s['sessions_majority_correct']} of {len(rows)} sessions called right by majority, "
          f"session AUC {s['session_auc_of_mean_p']:.2f} (exact p {s['session_auc_exact_p_two_sided']:.3f}), "
          f"prior-only pooled AUC {s['prior_only_pooled_auc']:.2f}")
    for name, r in rows.items():
        print(f"  {name:18s} queen={r['queen']} n={r['n']:4d} called queenright "
              f"{100 * r['called_queenright']:5.1f} %  median p {r['median_p']:.1e}")
    for name, r in out["wingbeat_per_session"].items():
        print(f"  {name:18s} 80-95 {r['mass_80_95']:.2f}  100-160 {r['mass_100_160']:.2f}  "
              f"180-260 {r['mass_180_260']:.2f}  p(100) {r['p_at_100']:.2f} x{r['line_ratio_100']:.1f}")
    for k, r in out["named_quantities_session_level"].items():
        print(f"  {k:10s} session AUC {r['session_auc']:.2f}  exact p {r['exact_p']:.2f} "
              f"({r['exact_p_without_CF001']:.2f} without CF001)")


if __name__ == "__main__":
    main()
