"""Fetch the 'To bee or not to bee' dataset from Zenodo (record 1321278).

Nolasco & Benetos, DCASE 2018, CC BY 4.0.  Two sources inside it: NU-Hive
(controlled, two hives, queen present / queen removed) and Open Source Beehives
(citizen science, several hives, 'Active' / 'Missing Queen').

Nothing is redistributed here; this downloads from the original DOI.  curl is
used rather than urllib because the python.org build on macOS has no CA bundle.
"""
from __future__ import annotations
import json, os, pathlib, subprocess, sys

# A local proxy is configured on this machine for other tooling; it does not
# route to Zenodo.  Clear it for everything this script spawns.
for _v in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
           "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_v, None)

RECORD = "1321278"
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw"


def curl(url: str) -> bytes:
    return subprocess.run(["curl", "-sL", url], capture_output=True, check=True).stdout


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads(curl(f"https://zenodo.org/api/records/{RECORD}"))
    files = sorted(meta["files"], key=lambda f: f["key"])
    print("%d files, %.2f GB" % (len(files), sum(f["size"] for f in files) / 1e9), flush=True)

    todo = [f for f in files
            if not ((OUT / f["key"]).exists() and (OUT / f["key"]).stat().st_size == f["size"])]
    print("%d still to fetch" % len(todo), flush=True)

    spec = "\n".join("url = %s\noutput = %s" % (f["links"]["self"], OUT / f["key"])
                     for f in todo)
    if spec:
        (ROOT / "data" / "_curl.cfg").write_text(spec)
        subprocess.run(["curl", "-sSL", "--parallel", "--parallel-max", "6",
                        "--retry", "3", "-K", str(ROOT / "data" / "_curl.cfg")], check=False)

    bad = [f["key"] for f in files
           if not (OUT / f["key"]).exists()
           or (OUT / f["key"]).stat().st_size != f["size"]]
    print("incomplete: %d" % len(bad), flush=True)
    for b in bad[:10]:
        print("  ", b, flush=True)


if __name__ == "__main__":
    main()
