"""What a pooled leave-one-out AUC is made of, read from the stored
out-of-fold probabilities (results/cv/).  No audio.

Section 4 reports one number per protocol.  Under leave-one-session-out
every held-out session carries one label, so the pooled AUC compares
clips scored by *different* models (nine of them) and is, exactly, a
clip-weighted mean of the twenty session-pair AUCs

    AUC_pooled = sum_ij n_i n_j AUC_ij / sum_ij n_i n_j,

one cell per (queenright session i, queenless session j), where AUC_ij is
the share of clip pairs in which the queenright clip scores higher.  This
script opens that sum, and asks three things the page could not:

  A. which cells carry the 0.054 and with what weight -- the six-clip
     session CF001 has 1/9 of the weight at the session level and
     0.2 % of it here; what the pooled number is without it, and what the
     unweighted mean of the twenty cells is;
  B. how much of the pull below 0.5 is the training prior (section 4.1:
     41-57 % queenright depending on which session is out).  Every fold's
     probabilities are shifted in logit space by minus the logit of that
     fold's training prior and the pooled AUC recomputed.  If the number
     barely moves, the prior is not what the network is doing;
  C. under leave-one-hive-out, how much of the 0.375 is the ordering
     *within* Hive1 and Hive3 (the only hives with both labels) and how
     much is which label each hive was handed: the pooled sum split into
     within-hive and cross-hive clip pairs.

The random-split folds (results/cv/random__fold*.json) are read as well
and the stored 0.9999 reproduced.  Writes results/pooled_decomposition.json.
Run from the repository root:

    python src/pooled_decomposition.py
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CV = RESULTS / "cv"


def auc(y, s):
    """Rank AUC with ties at half, as sklearn's roc_auc_score."""
    y = np.asarray(y, bool); s = np.asarray(s, float)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    allv = np.concatenate([pos, neg])
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pair_auc(a, b):
    """Share of (a_i, b_j) pairs with a_i > b_j, ties half."""
    a = np.asarray(a, float)[:, None]; b = np.asarray(b, float)[None, :]
    return float(((a > b).sum() + 0.5 * (a == b).sum()) / (a.size * b.size))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def session_protocol(meta):
    cv = {s: json.loads((CV / f"session__{s}.json").read_text()) for s in meta}
    n = {s: len(cv[s]["y"]) for s in meta}
    N = sum(n.values()); N1 = sum(n[s] for s in meta if meta[s]["queen"] == 1)
    qr = [s for s in meta if meta[s]["queen"] == 1]
    ql = [s for s in meta if meta[s]["queen"] == 0]
    P = {s: np.asarray(cv[s]["p"]) for s in meta}
    prior = {s: (N1 - n[s] * meta[s]["queen"]) / (N - n[s]) for s in meta}
    for s in meta:
        assert cv[s]["n_train"] == N - n[s]
        assert set(cv[s]["y"]) == {meta[s]["queen"]}

    # A. the twenty cells
    cells = {}
    for i in qr:
        for j in ql:
            cells[f"{i}|{j}"] = {"queenright": i, "queenless": j, "weight": n[i] * n[j],
                                 "auc": pair_auc(P[i], P[j])}
    W = sum(c["weight"] for c in cells.values())
    pooled = sum(c["weight"] * c["auc"] for c in cells.values()) / W
    yy = np.concatenate([np.full(n[s], meta[s]["queen"]) for s in meta])
    pp = np.concatenate([P[s] for s in meta])
    assert abs(pooled - auc(yy, pp)) < 1e-9
    for c in cells.values():
        c["weight_share"] = c["weight"] / W
    aucs = np.array([c["auc"] for c in cells.values()])
    no_cf = {k: c for k, c in cells.items() if c["queenless"] != "CF001"}
    W2 = sum(c["weight"] for c in no_cf.values())
    pooled_no_cf = sum(c["weight"] * c["auc"] for c in no_cf.values()) / W2
    # per session: the clip-weighted mean of its cells = its contribution
    per_session = {}
    for s in meta:
        mine = [c for c in cells.values() if s in (c["queenright"], c["queenless"])]
        w = sum(c["weight"] for c in mine)
        per_session[s] = {"queen": meta[s]["queen"], "n": n[s],
                          "weight_share_of_pooled": w / W,
                          "mean_auc_of_its_cells_clip_weighted": sum(c["weight"] * c["auc"] for c in mine) / w,
                          "mean_auc_of_its_cells_unweighted": float(np.mean([c["auc"] for c in mine])),
                          "cells_above_half": int(sum(c["auc"] > 0.5 for c in mine)),
                          "n_cells": len(mine)}

    # B. the prior pull
    L = {s: logit(P[s]) for s in meta}
    shifted = np.concatenate([L[s] - logit(prior[s]) for s in meta])
    centred = np.concatenate([L[s] - np.median(L[s]) for s in meta])
    prior_only = np.concatenate([np.full(n[s], prior[s]) for s in meta])
    b = {
        "train_prior_queenright": {s: prior[s] for s in meta},
        "pooled_auc": pooled,
        "pooled_auc_prior_only": auc(yy, prior_only),
        "pooled_auc_after_removing_train_prior": auc(yy, shifted),
        "shift_in_logit_min": float(min(abs(logit(prior[s])) for s in meta)),
        "shift_in_logit_max": float(max(abs(logit(prior[s])) for s in meta)),
        "median_abs_logit_of_network_min": float(min(np.median(abs(L[s])) for s in meta)),
        "median_abs_logit_of_network_max": float(max(np.median(abs(L[s])) for s in meta)),
        "pooled_auc_after_centring_each_session_at_its_median": auc(yy, centred),
    }
    # how big a *uniform* prior pull would it take: shift the queenright
    # folds down and queenless up by d logits until the pooled AUC reaches 0.5
    def pooled_at(d):
        z = np.concatenate([L[s] + (d if meta[s]["queen"] == 0 else -d) for s in meta])
        return auc(yy, z)
    grid = np.linspace(0, 40, 801)
    vals = [pooled_at(-d) for d in grid]        # undo a pull: push queenright up
    k = int(np.argmax(np.array(vals) >= 0.5))
    b["logit_shift_that_would_undo_the_reversal"] = float(grid[k]) if vals[k] >= 0.5 else None
    b["pooled_auc_if_prior_shift_were_undone_at_its_largest"] = pooled_at(-b["shift_in_logit_max"])

    return {
        "n_clips": N, "n_queenright_clips": N1, "n_cells": len(cells),
        "pooled_auc": pooled, "stored_pooled_auc": None,
        "cells": cells,
        "cell_auc_min": float(aucs.min()), "cell_auc_max": float(aucs.max()),
        "cell_auc_median": float(np.median(aucs)),
        "cells_above_half": int((aucs > 0.5).sum()),
        "cells_below_0_1": int((aucs < 0.1).sum()),
        "unweighted_mean_of_cells": float(aucs.mean()),
        "weight_share_CF001": sum(c["weight_share"] for c in cells.values() if c["queenless"] == "CF001"),
        "pooled_auc_without_CF001": pooled_no_cf,
        "per_session": per_session,
        "largest_cell_weight_share": max(c["weight_share"] for c in cells.values()),
        "prior": b,
    }


def hive_protocol(meta):
    hives = sorted({m["hive"] for m in meta.values()})
    cv = {h: json.loads((CV / f"hive__{h}.json").read_text()) for h in hives}
    Y = {h: np.asarray(cv[h]["y"], bool) for h in hives}
    P = {h: np.asarray(cv[h]["p"]) for h in hives}
    yy = np.concatenate([Y[h] for h in hives]); pp = np.concatenate([P[h] for h in hives])
    pooled = auc(yy, pp)
    # split the pooled sum: pairs (queenright clip, queenless clip) within a hive / across hives
    within_w = within_s = cross_w = cross_s = 0.0
    per_pair = {}
    for a in hives:
        for b in hives:
            pa, qb = P[a][Y[a]], P[b][~Y[b]]
            if len(pa) == 0 or len(qb) == 0:
                continue
            w = len(pa) * len(qb); s = pair_auc(pa, qb) * w
            per_pair[f"{a}|{b}"] = {"queenright_hive": a, "queenless_hive": b, "weight": w,
                                    "auc": s / w, "within_hive": a == b}
            if a == b:
                within_w += w; within_s += s
            else:
                cross_w += w; cross_s += s
    W = within_w + cross_w
    assert abs((within_s + cross_s) / W - pooled) < 1e-9
    for c in per_pair.values():
        c["weight_share"] = c["weight"] / W
    within = {h: cv[h].get("auc") for h in hives if len(set(cv[h]["y"])) == 2}
    return {
        "pooled_auc": pooled,
        "within_hive_pairs": {"weight_share": within_w / W, "auc": within_s / within_w,
                              "hives": within},
        "cross_hive_pairs": {"weight_share": cross_w / W, "auc": cross_s / cross_w},
        "pooled_if_within_hive_pairs_were_at_half": (0.5 * within_w + cross_s) / W,
        "pooled_if_cross_hive_pairs_were_at_half": (within_s + 0.5 * cross_w) / W,
        "per_pair": per_pair,
    }


def random_protocol():
    folds = sorted(CV.glob("random__fold*.json"))
    rows = [json.loads(f.read_text()) for f in folds]
    yy = np.concatenate([r["y"] for r in rows]); pp = np.concatenate([r["p"] for r in rows])
    return {"folds": len(rows), "n": int(len(yy)), "pooled_auc": auc(yy, pp),
            "per_fold_auc": [r["auc"] for r in rows],
            "per_fold_auc_recomputed": [auc(r["y"], r["p"]) for r in rows]}


def main():
    meta = json.loads((RESULTS / "session_meta.json").read_text())
    summary = json.loads((RESULTS / "protocol_summary.json").read_text())
    out = {"source": "results/cv", "session": session_protocol(meta),
           "hive": hive_protocol(meta), "random": random_protocol()}
    for k in ("session", "hive", "random"):
        out[k]["stored_pooled_auc"] = summary[k]["pooled_auc"]
        assert abs(out[k]["pooled_auc"] - summary[k]["pooled_auc"]) < 1e-9, k
    (RESULTS / "pooled_decomposition.json").write_text(json.dumps(out, indent=1))

    s = out["session"]
    print(f"session: pooled {s['pooled_auc']:.4f} (stored {s['stored_pooled_auc']:.4f}); "
          f"{s['n_cells']} cells {s['cell_auc_min']:.4f}-{s['cell_auc_max']:.4f}, median "
          f"{s['cell_auc_median']:.4f}, unweighted mean {s['unweighted_mean_of_cells']:.4f}, "
          f"{s['cells_above_half']} above 0.5, {s['cells_below_0_1']} below 0.1; CF001 weight "
          f"{100 * s['weight_share_CF001']:.2f} %, pooled without it {s['pooled_auc_without_CF001']:.4f}; "
          f"largest cell weight {100 * s['largest_cell_weight_share']:.1f} %")
    head = "queenright / queenless"
    print(f"{head:26s}" + "".join(f"{j[:14]:>16s}" for j in
          [c['queenless'] for c in list(s['cells'].values())[:5]]))
    qr = sorted({c["queenright"] for c in s["cells"].values()})
    ql = sorted({c["queenless"] for c in s["cells"].values()})
    for i in qr:
        print(f"{i:26s}" + "".join(f"{s['cells'][f'{i}|{j}']['auc']:16.4f}" for j in ql))
    for name, r in s["per_session"].items():
        print(f"  {name:20s} q={r['queen']} n={r['n']:4d} weight {100 * r['weight_share_of_pooled']:5.1f} % "
              f"cells {r['mean_auc_of_its_cells_clip_weighted']:.4f} / unweighted "
              f"{r['mean_auc_of_its_cells_unweighted']:.4f}, {r['cells_above_half']}/{r['n_cells']} above 0.5")
    b = s["prior"]
    print(f"prior: prior-only {b['pooled_auc_prior_only']:.4f}; after removing the training prior "
          f"{b['pooled_auc_after_removing_train_prior']:.4f}; prior shifts {b['shift_in_logit_min']:.3f}-"
          f"{b['shift_in_logit_max']:.3f} logits vs network |logit| medians "
          f"{b['median_abs_logit_of_network_min']:.1f}-{b['median_abs_logit_of_network_max']:.1f}; "
          f"undo at largest {b['pooled_auc_if_prior_shift_were_undone_at_its_largest']:.4f}; shift to "
          f"reach 0.5: {b['logit_shift_that_would_undo_the_reversal']}; centred at medians "
          f"{b['pooled_auc_after_centring_each_session_at_its_median']:.4f}")
    h = out["hive"]
    print(f"hive: pooled {h['pooled_auc']:.4f}; within-hive pairs {100 * h['within_hive_pairs']['weight_share']:.1f} % "
          f"of the weight at {h['within_hive_pairs']['auc']:.4f} ({h['within_hive_pairs']['hives']}); cross-hive "
          f"{100 * h['cross_hive_pairs']['weight_share']:.1f} % at {h['cross_hive_pairs']['auc']:.4f}; "
          f"within at half -> {h['pooled_if_within_hive_pairs_were_at_half']:.4f}, cross at half -> "
          f"{h['pooled_if_cross_hive_pairs_were_at_half']:.4f}")
    for k, c in h["per_pair"].items():
        print(f"  {k:16s} {'within' if c['within_hive'] else 'cross ':6s} weight {100 * c['weight_share']:5.1f} % auc {c['auc']:.4f}")
    r = out["random"]
    print(f"random: pooled {r['pooled_auc']:.6f} (stored {r['stored_pooled_auc']:.6f}), per fold "
          f"{[round(v, 5) for v in r['per_fold_auc_recomputed']]}")


if __name__ == "__main__":
    main()
