"""Pins section 4.2 ("What the pooled number is made of") to
results/pooled_decomposition.json (written by src/pooled_decomposition.py
from results/cv/) and recomputes the cheap parts from results/cv/ itself.
No audio is needed.
"""

import json
import os
import sys
from decimal import ROUND_HALF_UP, Decimal

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
README = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
FLAT = " ".join(README.split())
PD = json.load(open(os.path.join(ROOT, "results", "pooled_decomposition.json")))
PS = json.load(open(os.path.join(ROOT, "results", "protocol_summary.json")))
META = json.load(open(os.path.join(ROOT, "results", "session_meta.json")))

QR = ["CF003", "GH001", "Hive1_12_06_2018", "Hive3_20_07_2017"]
QL = ["CF001", "CJ001", "Hive1_31_05_2018", "Hive3_12_07_2017", "Hive3_15_07_2017"]
NAME = {"CF001": "CF001", "CF003": "CF003", "CJ001": "CJ001", "GH001": "GH001",
        "Hive1_12_06_2018": "Hive1, 12 June", "Hive1_31_05_2018": "Hive1, 31 May",
        "Hive3_12_07_2017": "Hive3, 12 July", "Hive3_15_07_2017": "Hive3, 15 July",
        "Hive3_20_07_2017": "Hive3, 20 July"}
HIVES_QR = ["CF003", "GH001", "Hive1", "Hive3"]
HIVES_QL = ["CF001", "CJ001", "Hive1", "Hive3"]


def r(x, nd=2):
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def cell_row(i):
    c = PD["session"]["cells"]
    return (f"| {NAME[i]} | " + " | ".join(r(c[f"{i}|{j}"]["auc"], 3) for j in QL) + " |")


def hive_row(a):
    pp = PD["hive"]["per_pair"]
    return (f"| {a} | " + " | ".join(
        f"{r(pp[f'{a}|{b}']['auc'], 2)} ({r(100 * pp[f'{a}|{b}']['weight_share'], 1)} %)"
        for b in HIVES_QL) + " |")


def test_pooled_aucs_are_the_stored_ones():
    for k in ("session", "hive", "random"):
        assert abs(PD[k]["pooled_auc"] - PS[k]["pooled_auc"]) < 1e-9
        assert abs(PD[k]["stored_pooled_auc"] - PS[k]["pooled_auc"]) < 1e-12


def test_cells_recomputed_from_cv_and_sum_to_the_pooled_auc():
    from pooled_decomposition import pair_auc
    cv = {s: json.load(open(os.path.join(ROOT, "results", "cv", f"session__{s}.json")))
          for s in META}
    W = tot = 0.0
    for i in QR:
        for j in QL:
            a = pair_auc(cv[i]["p"], cv[j]["p"])
            c = PD["session"]["cells"][f"{i}|{j}"]
            assert abs(a - c["auc"]) < 1e-12, (i, j)
            w = len(cv[i]["p"]) * len(cv[j]["p"])
            assert c["weight"] == w
            W += w; tot += w * a
    assert abs(tot / W - PS["session"]["pooled_auc"]) < 1e-9


def test_cell_table_rows_and_summary():
    s = PD["session"]
    for i in QR:
        assert cell_row(i) in README, cell_row(i)
    assert s["n_cells"] == 20 and s["cells_above_half"] == 0
    assert s["cells_below_0_1"] == 16
    assert r(s["cell_auc_max"], 2) == "0.29"
    assert s["cells"]["GH001|Hive1_31_05_2018"]["auc"] == s["cell_auc_max"]
    assert r(s["cell_auc_median"], 3) == "0.008"
    assert r(s["unweighted_mean_of_cells"], 3) == "0.051"
    assert r(100 * s["weight_share_CF001"], 1) == "0.3"
    assert r(s["pooled_auc_without_CF001"], 4) == r(s["pooled_auc"], 4) == "0.0536"
    assert r(100 * s["largest_cell_weight_share"], 1) == "7.8"
    for t in ("None of the 20 is above 0.5", "16 of them are below 0.1", "0.29", "0.3 %", "0.051"):
        assert t in FLAT, t
    # the three OSBH-vs-NU-Hive cells that carry the only ordering worth the name
    assert r(s["cells"]["CF003|Hive1_31_05_2018"]["auc"], 2) == "0.20"
    assert r(s["cells"]["Hive1_12_06_2018|Hive3_12_07_2017"]["auc"], 4) == "0.0000"
    c = s["cells"]
    nu = [c[f"{i}|{j}"]["auc"] for i in ("Hive1_12_06_2018", "Hive3_20_07_2017") for j in QL]
    assert max(nu) < 0.03                                   # "more than 97 %"
    osbh = [c[f"{i}|{j}"]["auc"] for i in ("CF003", "GH001")
            for j in ("CF001", "CJ001", "Hive1_31_05_2018")]
    assert r(100 * min(osbh), 0) == "8" and r(100 * max(osbh), 0) == "29"
    h3 = [c[f"{i}|{j}"]["auc"] for i in ("CF003", "GH001")
          for j in ("Hive3_12_07_2017", "Hive3_15_07_2017")]
    assert r(100 * min(h3), 0) == "0" and r(100 * max(h3), 0) == "3"
    assert "8 to 29 %" in FLAT and "0 to 3 %" in FLAT and "more than 97 %" in FLAT


def test_the_training_prior_is_not_the_pull():
    b = PD["session"]["prior"]
    assert b["pooled_auc_prior_only"] == 0.0
    assert r(b["pooled_auc_after_removing_train_prior"], 3) == "0.064"
    assert r(b["shift_in_logit_min"], 2) == "0.02" and r(b["shift_in_logit_max"], 2) == "0.36"
    assert r(b["median_abs_logit_of_network_min"], 1) == "2.5"
    assert r(b["median_abs_logit_of_network_max"], 1) == "14.9"
    assert r(b["logit_shift_that_would_undo_the_reversal"], 1) == "6.8"
    assert r(b["pooled_auc_if_prior_shift_were_undone_at_its_largest"], 3) == "0.066"
    assert r(b["pooled_auc_after_centring_each_session_at_its_median"], 2) == "0.50"
    priors = b["train_prior_queenright"]
    qr = sorted(priors[s] for s in QR); ql = sorted(priors[s] for s in QL)
    assert r(100 * qr[0], 0) == "41" and r(100 * qr[-1], 0) == "45"
    assert r(100 * ql[0], 0) == "49" and r(100 * ql[-1], 0) == "57"
    for t in ("0.054 to 0.064", "6.8 logits", "0.02 to 0.36"):
        assert t in FLAT, t


def test_hive_protocol_split():
    h = PD["hive"]
    assert r(100 * h["within_hive_pairs"]["weight_share"], 0) == "16"
    assert r(h["within_hive_pairs"]["auc"], 2) == "0.43"
    assert r(h["cross_hive_pairs"]["auc"], 2) == "0.36"
    assert r(h["pooled_if_cross_hive_pairs_were_at_half"], 2) == "0.49"
    assert r(h["pooled_if_within_hive_pairs_were_at_half"], 2) == "0.39"
    assert r(h["within_hive_pairs"]["hives"]["Hive1"], 2) == "0.49"
    assert r(h["within_hive_pairs"]["hives"]["Hive3"], 2) == "0.38"
    for a in HIVES_QR:
        assert hive_row(a) in README, hive_row(a)
    pp = h["per_pair"]
    assert r(pp["CF003|Hive1"]["auc"], 2) == "0.99" and r(pp["GH001|Hive1"]["auc"], 2) == "0.99"
    assert r(pp["CF003|Hive3"]["auc"], 2) == "0.15" and r(pp["Hive1|Hive3"]["auc"], 3) == "0.001"
    assert r(pp["Hive3|Hive1"]["auc"], 3) == "0.999"
    assert r(100 * (1 - pp["CF003|Hive3"]["auc"]), 0) == "85"
    assert r(100 * (1 - pp["GH001|Hive3"]["auc"]), 0) == "86"
    assert "85 to 86 %" in FLAT
    assert "16 % of the pooled weight" in FLAT and "0.49" in FLAT


def test_random_folds_are_published_and_reproduce():
    import glob
    files = sorted(glob.glob(os.path.join(ROOT, "results", "cv", "random__fold*.json")))
    assert len(files) == 5
    assert len(glob.glob(os.path.join(ROOT, "results", "cv", "*.json"))) == 20
    rr = PD["random"]
    for a, b in zip(rr["per_fold_auc"], rr["per_fold_auc_recomputed"]):
        assert abs(a - b) < 1e-9
    assert r(rr["pooled_auc"], 4) == "0.9999"


if __name__ == "__main__":
    print("| queenright \\ queenless | " + " | ".join(NAME[j] for j in QL) + " |")
    for i in QR:
        print(cell_row(i))
    print()
    print("| queenright \\ queenless | " + " | ".join(HIVES_QL) + " |")
    for a in HIVES_QR:
        print(hive_row(a))
    b = PD["session"]["prior"]
    print(b["train_prior_queenright"])
