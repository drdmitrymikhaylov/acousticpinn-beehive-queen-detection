"""Turn the Zenodo recordings into labelled clips, keeping provenance.

Two label sources, and they matter for how the data may be split:

* NU-Hive: two hives recorded on separate days with the queen present and with
  the queen removed.  The queen status is in the file name, and so is the
  recording session.
* Open Source Beehives: several citizen-science hives, each labelled 'Active'
  or 'Missing Queen'.  There is no session structure; the hive is the unit.

Only segments the dataset annotates as 'bee' are kept, so that the classifier
is not rewarded for hearing a passing car.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLIPS = ROOT / "data" / "clips"
SR = 8000
# Two seconds, not five: many Open Source Beehives segments between two
# external noises are shorter than five seconds, and the colony hum is
# stationary enough that two seconds still gives 0.5 Hz resolution.
CLIP_S = 2.0


def parse_meta(stem: str) -> dict | None:
    """Hive, recording session and queen status, from the file name."""
    if stem.startswith("Hive"):
        m = re.match(r"(Hive\d+)_(\d{2}_\d{2}_\d{4})_(NO_)?QueenBee_", stem)
        if not m:
            return None
        hive, date, no_queen = m.group(1), m.group(2), m.group(3)
        return {"source": "NU-Hive", "hive": hive, "session": f"{hive}_{date}",
                "queen": 0 if no_queen else 1}
    if " - " in stem:
        hive = stem.split(" - ")[0].strip()
        queen = 0 if "Missing Queen" in stem else 1
        return {"source": "OSBH", "hive": hive, "session": hive, "queen": queen}
    return None


def read_lab(path: pathlib.Path):
    """(start, end) of every segment the annotators marked as hive sound."""
    out = []
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.replace(",", ".").split()
        if len(parts) < 3:
            continue
        try:
            a, b = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        if parts[2].strip().lower() == "bee":
            out.append((a, b))
    return out


def load_mono(path: pathlib.Path):
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
    except Exception:
        # libsndfile without mpeg support: decode through ffmpeg instead
        p = subprocess.run(
            ["ffmpeg", "-v", "quiet", "-i", str(path), "-f", "f32le",
             "-ac", "1", "-ar", str(SR), "-"],
            capture_output=True, check=True)
        return np.frombuffer(p.stdout, dtype=np.float32).copy(), SR
    return x, sr


def to_sr(x, sr):
    if sr == SR:
        return x
    from math import gcd
    g = gcd(int(sr), SR)
    return resample_poly(x, SR // g, int(sr) // g).astype(np.float32)


def complete_files():
    """Only files whose size matches the Zenodo manifest.

    The download runs in the background, so a half-written file is the normal
    state, not an exception.  Reading one gives a short signal against a full
    annotation file and silently throws most of the data away.
    """
    rec = json.loads((ROOT / "data" / "rec.json").read_text())
    keep = []
    for f in rec["files"]:
        p = RAW / f["key"]
        if p.suffix.lower() not in (".wav", ".mp3"):
            continue
        if p.exists() and p.stat().st_size == f["size"]:
            keep.append(p)
    return sorted(keep)


def main():
    CLIPS.mkdir(parents=True, exist_ok=True)
    index = []
    audio = complete_files()
    for i, p in enumerate(audio):
        meta = parse_meta(p.stem)
        lab = p.with_suffix(".lab")
        if meta is None or not lab.exists():
            print("skip", p.name, flush=True)
            continue
        out = CLIPS / (p.stem + ".npz")
        if out.exists():
            z = np.load(out, allow_pickle=True)
            index.append({**meta, "file": p.stem, "n_clips": int(z["X"].shape[0])})
            continue

        segs = read_lab(lab)
        if not segs:
            print("no bee segments", p.name, flush=True)
            continue
        x, sr = load_mono(p)
        x = to_sr(x, sr)
        n = int(CLIP_S * SR)
        clips = []
        for a, b in segs:
            i0, i1 = int(a * SR), min(int(b * SR), x.size)
            for s in range(i0, i1 - n + 1, n):
                c = x[s:s + n]
                if np.max(np.abs(c)) < 1e-6:
                    continue
                clips.append(c / (np.max(np.abs(c)) + 1e-9))
        if not clips:
            continue
        X = np.stack(clips).astype(np.float32)
        np.savez_compressed(out, X=X)
        index.append({**meta, "file": p.stem, "n_clips": int(X.shape[0])})
        print("[%3d/%3d] %-55s %4d clips  queen=%d  %s"
              % (i + 1, len(audio), p.stem[:55], X.shape[0], meta["queen"],
                 meta["session"]), flush=True)

    (ROOT / "data" / "index.json").write_text(json.dumps(index, indent=2))
    tot = sum(r["n_clips"] for r in index)
    print("\n%d files, %d clips, %.1f h"
          % (len(index), tot, tot * CLIP_S / 3600), flush=True)


if __name__ == "__main__":
    main()
