"""Figures.  Run after prepare.py, spectra.py and train.py."""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                    # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
RES = ROOT / "results"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, SURFACE = "#1a1a1a", "#7a7a7a", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9,
    "axes.edgecolor": "#d8d8d4", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": "#e8e8e4", "grid.linewidth": 0.7, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 160,
})


def fig_spectra():
    z = np.load(RES / "session_psd.npz")
    meta = json.loads((RES / "session_meta.json").read_text())
    f = z["f"]
    sources = sorted({m["source"] for m in meta.values()})
    fig, axes = plt.subplots(1, len(sources), figsize=(4.6 * len(sources), 3.4),
                             squeeze=False)
    for ax, src in zip(axes[0], sources):
        for sess, m in meta.items():
            if m["source"] != src:
                continue
            P = z[sess]
            P = P / np.trapezoid(P[(f >= 20) & (f <= 4000)], f[(f >= 20) & (f <= 4000)])
            c = BLUE if m["queen"] == 1 else ORANGE
            ax.semilogx(f[1:], 10 * np.log10(P[1:] + 1e-20), color=c, lw=1.4,
                        alpha=0.85, label=None)
        ax.axvspan(180, 260, color=AQUA, alpha=0.10)
        ax.text(200, ax.get_ylim()[1], " worker flight\n fundamental",
                fontsize=7, color=AQUA, va="top")
        ax.axvspan(500, 600, color=MUTED, alpha=0.10)
        ax.set_xlim(20, 4000)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("power spectral density (dB, area-normalised)")
        ax.set_title("%s  (blue = queenright, orange = queenless)" % src,
                     loc="left", fontsize=9.5)
    fig.suptitle("Every recording session's average spectrum",
                 x=0.01, ha="left", fontsize=11, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "01_session_spectra.png"); plt.close(fig)


def fig_bands():
    rows = json.loads((RES / "band_table.json").read_text())
    bands = [k for k in rows[0] if k not in
             ("session", "hive", "source", "queen", "n_clips")]
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    x = np.arange(len(bands))
    for r in rows:
        c = BLUE if r["queen"] == 1 else ORANGE
        ax.plot(x, [100 * r[b] for b in bands], "o-", color=c, lw=1.1, ms=3.5,
                alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([b.split(" (")[0] for b in bands], rotation=25, ha="right",
                       fontsize=8)
    ax.set_ylabel("share of 20 Hz - 4 kHz power (%)")
    ax.plot([], [], "o-", color=BLUE, label="queenright session")
    ax.plot([], [], "o-", color=ORANGE, label="queenless session")
    ax.legend(fontsize=8)
    ax.set_title("One line per recording session, not per clip\n"
                 "the spread between sessions of the same class is the "
                 "honest error bar", loc="left", fontsize=9.5)
    fig.tight_layout(); fig.savefig(FIG / "02_band_shares.png"); plt.close(fig)


def fig_protocols():
    d = json.loads((RES / "protocol_summary.json").read_text())
    order = [p for p in ("random", "session", "hive") if p in d]
    labels = {"random": "random split\nover clips",
              "session": "leave one\nrecording session out",
              "hive": "leave one\nhive out"}
    colors = {"random": ORANGE, "session": BLUE, "hive": AQUA}
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for i, p in enumerate(order):
        v = d[p]["pooled_auc"]
        ax.bar(i, v, 0.55, color=colors[p])
        ax.text(i, v + 0.015, "%.3f" % v, ha="center", fontsize=9, color=INK)
    ax.axhline(0.5, color=MUTED, ls="--", lw=1)
    ax.text(len(order) - 0.5, 0.515, "chance", fontsize=8, color=MUTED, ha="right")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([labels[p] for p in order], fontsize=8.5)
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("pooled ROC AUC, queen present vs absent")
    ax.set_title("The same data, the same model, three ways of splitting it",
                 loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "03_protocols.png"); plt.close(fig)



def fig_wingbeat():
    d = json.loads((RES / "wingbeat_pinn.json").read_text())
    f0 = np.array(d["f0_grid"]); f = np.array(d["f"])
    prof = np.array(d["harmonic_profile"])              # (H, K)
    per = d["per_session"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.6),
                             gridspec_kw={"width_ratios": [1.15, 1, 1]})

    ax = axes[0]
    for sess, m in per.items():
        c = BLUE if m["queen"] == 1 else ORANGE
        ax.plot(f0, m["p_f0"], color=c, lw=1.3, alpha=0.85)
    ax.set_xlabel("wing-beat fundamental f0 (Hz)")
    ax.set_ylabel("colony share p(f0), session mean")
    ax.axvspan(180, 260, color=AQUA, alpha=0.10)
    ax.text(182, ax.get_ylim()[1] * 0.97, "textbook worker\nflight 180-260 Hz",
            fontsize=7, color=AQUA, va="top")
    ax.set_title("where the model puts the wing beats\n(blue queenright, orange queenless)",
                 loc="left", fontsize=9.5)

    ax = axes[1]
    k = len(f0) // 2
    ax.plot(f, np.array(d["comb_basis_example"]), color=INK, lw=1.0)
    ax.set_xlim(20, 1500); ax.set_yscale("log")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("one bee at f0 = %.0f Hz" % f0[k])
    ax.set_title("the learned single-bee spectrum: harmonics of one stroke,\n"
                 "line width %.1f Hz x harmonic number" % d["gamma_hz"],
                 loc="left", fontsize=9.5)

    ax = axes[2]
    order = ["random", "session", "hive"]
    labels = {"random": "random\nover clips", "session": "leave one\nsession out",
              "hive": "leave one\nhive out"}
    keys = [("wingbeat_5_features", "wing-beat model, 5 named numbers", AQUA),
            ("wingbeat_distribution", "wing-beat model, full p(f0)", BLUE),
            ("band_energies", "8 band energies", MUTED)]
    x = np.arange(len(order)); w = 0.2
    for j, (key, lab, col) in enumerate(keys):
        v = [d["protocols"][p][key]["pooled_auc"] for p in order]
        ax.bar(x + (j - 1.5) * w, v, w, color=col, label=lab)
    cnn = d.get("cnn_pooled_auc", {})
    if cnn:
        v = [cnn.get(p, np.nan) for p in order]
        ax.bar(x + 1.5 * w, v, w, color=ORANGE, label="log-mel CNN (train.py)")
    ax.axhline(0.5, color=MUTED, ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([labels[p] for p in order], fontsize=8.5)
    ax.set_ylim(0.0, 1.05); ax.set_ylabel("pooled ROC AUC")
    ax.legend(fontsize=6.5, loc="lower left")
    ax.set_title("what each reading of the sound is worth\nunder each protocol",
                 loc="left", fontsize=9.5)
    fig.tight_layout(); fig.savefig(FIG / "04_wingbeat_model.png"); plt.close(fig)


if __name__ == "__main__":
    for fn in (fig_spectra, fig_bands, fig_protocols, fig_wingbeat):
        try:
            fn()
        except FileNotFoundError as e:
            print("skip %s (%s)" % (fn.__name__, e))
    print("figures written")
