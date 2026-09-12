"""Queen presence from hive sound, under three splitting protocols.

The point of this file is the comparison, not the best number.

  random   -- five stratified folds over clips, ignoring where a clip came
              from.  Clips from the same ten-minute recording land on both
              sides of the split.  This is how much of the published
              literature evaluates, and it is the control condition here.
  session  -- leave one recording session out.
  hive     -- leave one hive out.  The only protocol that answers the question
              a beekeeper would ask: will this work on MY colony.

Folds under 'session' and 'hive' often contain a single class, so a per-fold
AUC is undefined.  Out-of-fold scores are therefore pooled and scored once,
which is the correct way to read a leave-one-group-out experiment.

Resumable: one JSON per (protocol, fold).
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from model import BeeNet, SR                       # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLIPS = ROOT / "data" / "clips"
RESULTS = ROOT / "results" / "cv"
RESULTS.mkdir(parents=True, exist_ok=True)

MAX_CLIPS_PER_SESSION = 600
EPOCHS = 14
BATCH = 32


def device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def load_all(seed=0):
    """Clips plus the group labels the protocols split on."""
    rng = np.random.default_rng(seed)
    index = json.loads((ROOT / "data" / "index.json").read_text())
    by_session = {}
    for r in index:
        by_session.setdefault(r["session"], []).append(r)

    X, y, sess, hive = [], [], [], []
    for s, items in sorted(by_session.items()):
        chunks = [np.load(CLIPS / (r["file"] + ".npz"))["X"] for r in items]
        A = np.concatenate(chunks)
        if A.shape[0] > MAX_CLIPS_PER_SESSION:
            A = A[rng.choice(A.shape[0], MAX_CLIPS_PER_SESSION, replace=False)]
        X.append(A)
        y += [items[0]["queen"]] * A.shape[0]
        sess += [s] * A.shape[0]
        hive += [items[0]["hive"]] * A.shape[0]
    return (np.concatenate(X), np.array(y, dtype=np.int64),
            np.array(sess), np.array(hive))


def fit(model, X, y, tr, dev, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    counts = np.array([(y[tr] == c).sum() for c in (0, 1)], dtype=float)
    w = torch.tensor(counts.sum() / np.maximum(counts, 1), dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=(w / w.sum() * 2).to(dev))
    Xtr = torch.from_numpy(X[tr]); ytr = torch.from_numpy(y[tr])
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(Xtr.size(0))
        for i in range(0, perm.numel(), BATCH):
            b = perm[i:i + BATCH]
            xb = torch.from_numpy(
                np.roll(Xtr[b].numpy(), int(rng.integers(-SR, SR)), axis=-1)).to(dev)
            opt.zero_grad()
            lossf(model(xb), ytr[b].to(dev)).backward()
            opt.step()
        sched.step()


def predict(model, X, te, dev):
    model.eval()
    Xte = torch.from_numpy(X[te])
    out = []
    with torch.no_grad():
        for i in range(0, Xte.size(0), 64):
            lg = model(Xte[i:i + 64].to(dev))
            out.append(torch.softmax(lg, 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def folds_for(protocol, y, sess, hive):
    if protocol == "random":
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        return [(f"fold{i}", tr, te) for i, (tr, te) in enumerate(skf.split(y, y))]
    groups = sess if protocol == "session" else hive
    out = []
    for g in sorted(set(groups)):
        te = np.flatnonzero(groups == g)
        tr = np.flatnonzero(groups != g)
        if len(set(y[tr])) < 2:
            continue
        out.append((g, tr, te))
    return out


def main():
    dev = device()
    X, y, sess, hive = load_all()
    print("clips %d  queenright %d  queenless %d  sessions %d  hives %d"
          % (len(y), (y == 1).sum(), (y == 0).sum(),
             len(set(sess)), len(set(hive))), flush=True)

    for protocol in ("random", "session", "hive"):
        for name, tr, te in folds_for(protocol, y, sess, hive):
            out = RESULTS / f"{protocol}__{name}.json".replace("/", "_")
            if out.exists():
                continue
            model = BeeNet().to(dev)
            fit(model, X, y, tr, dev, seed=abs(hash((protocol, name))) % 10_000)
            p = predict(model, X, te, dev)
            rec = {"protocol": protocol, "fold": name,
                   "n_train": int(len(tr)), "n_test": int(len(te)),
                   "test_classes": sorted(set(int(v) for v in y[te])),
                   "y": [int(v) for v in y[te]], "p": [float(v) for v in p]}
            if len(rec["test_classes"]) == 2:
                rec["auc"] = float(roc_auc_score(y[te], p))
            out.write_text(json.dumps(rec))
            print("%-8s %-20s n_test=%5d %s" %
                  (protocol, name, len(te),
                   ("auc %.3f" % rec["auc"]) if "auc" in rec else "single-class fold"),
                  flush=True)

    # pooled scoring
    summary = {}
    for protocol in ("random", "session", "hive"):
        rows = [json.loads(f.read_text())
                for f in sorted(RESULTS.glob(f"{protocol}__*.json"))]
        if not rows:
            continue
        yy = np.concatenate([r["y"] for r in rows])
        pp = np.concatenate([r["p"] for r in rows])
        summary[protocol] = {
            "folds": len(rows), "n": int(yy.size),
            "pooled_auc": float(roc_auc_score(yy, pp)),
            "pooled_bal_acc": float(balanced_accuracy_score(yy, (pp >= 0.5).astype(int))),
        }
        per = [r["auc"] for r in rows if "auc" in r]
        if per:
            summary[protocol]["per_fold_auc_mean"] = float(np.mean(per))
            summary[protocol]["per_fold_auc_sd"] = float(np.std(per))
    (ROOT / "results" / "protocol_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
