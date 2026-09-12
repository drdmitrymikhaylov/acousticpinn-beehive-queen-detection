"""What a hive actually sounds like, measured rather than asserted.

The literature names a handful of bands and attaches meanings to them.  This
module measures the average spectrum of every recording session in the dataset
and reports the band energies per session, split by queen status.

The unit of replication is the SESSION, not the clip.  Ten thousand five-second
clips from one afternoon in one hive are one observation of one colony, and
treating them as ten thousand independent samples is how a difference between
two recordings becomes a p-value with ten zeros in it.  Everything below is
therefore aggregated to the session first, and the session-level spread is
reported next to the mean.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy.signal import welch

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLIPS = ROOT / "data" / "clips"
RESULTS = ROOT / "results"
SR = 8000

# Bands named in the literature; see docs/references.md for who reported what.
BANDS = {
    "20-100 Hz (quiet colony)": (20, 100),
    "100-180 Hz (low activity)": (100, 180),
    "180-260 Hz (worker flight fundamental)": (180, 260),
    "260-400 Hz": (260, 400),
    "400-500 Hz": (400, 500),
    "500-600 Hz (reported swarming band)": (500, 600),
    "600-1000 Hz": (600, 1000),
    "1-4 kHz": (1000, 4000),
}


def load_index():
    return json.loads((ROOT / "data" / "index.json").read_text())


def session_psd(rows, max_clips=1200, seed=0):
    """Mean PSD per session, and the metadata that goes with it."""
    rng = np.random.default_rng(seed)
    by = {}
    for r in rows:
        by.setdefault(r["session"], []).append(r)

    out = {}
    for sess, items in sorted(by.items()):
        X = []
        for r in items:
            z = np.load(CLIPS / (r["file"] + ".npz"))
            X.append(z["X"])
        X = np.concatenate(X)
        if X.shape[0] > max_clips:
            X = X[rng.choice(X.shape[0], max_clips, replace=False)]
        f, P = welch(X, fs=SR, nperseg=2048, noverlap=1024, axis=-1)
        out[sess] = {"f": f, "psd": P.mean(axis=0),
                     "queen": items[0]["queen"], "source": items[0]["source"],
                     "hive": items[0]["hive"], "n_clips": int(X.shape[0])}
    return out


def band_table(psd_by_session):
    """Share of total 20 Hz - 4 kHz power falling in each named band."""
    rows = []
    for sess, d in psd_by_session.items():
        f, P = d["f"], d["psd"]
        total = np.trapezoid(P[(f >= 20) & (f <= 4000)], f[(f >= 20) & (f <= 4000)])
        rec = {"session": sess, "hive": d["hive"], "source": d["source"],
               "queen": d["queen"], "n_clips": d["n_clips"]}
        for name, (lo, hi) in BANDS.items():
            m = (f >= lo) & (f < hi)
            rec[name] = float(np.trapezoid(P[m], f[m]) / total)
        rows.append(rec)
    return rows


def main():
    RESULTS.mkdir(exist_ok=True)
    index = load_index()
    psd = session_psd(index)

    np.savez_compressed(RESULTS / "session_psd.npz",
                        **{k: v["psd"] for k, v in psd.items()},
                        f=next(iter(psd.values()))["f"])
    meta = {k: {kk: vv for kk, vv in v.items() if kk not in ("f", "psd")}
            for k, v in psd.items()}
    (RESULTS / "session_meta.json").write_text(json.dumps(meta, indent=2))

    rows = band_table(psd)
    (RESULTS / "band_table.json").write_text(json.dumps(rows, indent=2))

    print("%-22s %-8s %-6s %s" % ("session", "source", "queen", "top band"))
    for r in rows:
        bands = {k: v for k, v in r.items() if k in BANDS}
        top = max(bands, key=bands.get)
        print("%-22s %-8s %-6d %s (%.0f %%)"
              % (r["session"], r["source"], r["queen"], top, 100 * bands[top]))

    # Session-level contrast, reported with the spread across sessions rather
    # than a p-value over clips.
    print("\nband share, mean over sessions (n sessions in brackets)")
    print("%-42s %14s %14s" % ("band", "queenright", "queenless"))
    summary = {}
    for name in BANDS:
        a = [r[name] for r in rows if r["queen"] == 1]
        b = [r[name] for r in rows if r["queen"] == 0]
        summary[name] = {"queenright_mean": float(np.mean(a)),
                         "queenright_sd": float(np.std(a)), "n_queenright": len(a),
                         "queenless_mean": float(np.mean(b)),
                         "queenless_sd": float(np.std(b)), "n_queenless": len(b)}
        print("%-42s %6.1f +/- %-5.1f %6.1f +/- %-5.1f"
              % (name, 100 * np.mean(a), 100 * np.std(a),
                 100 * np.mean(b), 100 * np.std(b)))
    (RESULTS / "band_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
