"""Pin sections 4.1 and 5.1 of the README to the stored outputs.

Needs no audio and no torch: only results/cv/ (out-of-fold probabilities of
the network), results/wingbeat_pinn.json and results/session_checks.json.

    python -m pytest tests/test_readme_numbers.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import session_checks as sc  # noqa: E402

RESULTS = ROOT / "results"
CHK = json.loads((RESULTS / "session_checks.json").read_text())
META = json.loads((RESULTS / "session_meta.json").read_text())
WB = json.loads((RESULTS / "wingbeat_pinn.json").read_text())
README = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())

NAME = {
    "CF001": "CF001", "CF003": "CF003", "CJ001": "CJ001", "GH001": "GH001",
    "Hive1_31_05_2018": "Hive1, 31 May", "Hive1_12_06_2018": "Hive1, 12 June",
    "Hive3_12_07_2017": "Hive3, 12 July", "Hive3_15_07_2017": "Hive3, 15 July",
    "Hive3_20_07_2017": "Hive3, 20 July",
}
SESS = CHK["cnn_session_protocol"]["per_session"]
SUMM = CHK["cnn_session_protocol"]["summary"]
HIVE = CHK["cnn_hive_protocol"]
WING = CHK["wingbeat_per_session"]
NAMED = CHK["named_quantities_session_level"]


def pct(x):
    v = 100.0 * x
    if v == 0 or v == 100 or 1.0 <= v <= 99.0:
        return "%.0f %%" % v
    return "%.1f %%" % v


def prob(p):
    return "%.0e" % p if p < 0.005 else "%.2f" % p


def close(a, b, tol=1e-9):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k], tol) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return abs(a - b) <= tol
    return a == b


def test_stored_json_is_what_the_script_computes():
    rows, summary = sc.cnn_by_session(META)
    assert close(rows, SESS) and close(summary, SUMM)
    assert close(sc.cnn_by_hive(META), HIVE)
    assert close(sc.wingbeat_by_session(WB), WING)
    assert close(sc.named_quantities(WB), NAMED)


def test_pooled_auc_is_the_one_in_section_4():
    summary = json.loads((RESULTS / "protocol_summary.json").read_text())
    assert abs(SUMM["pooled_auc"] - summary["session"]["pooled_auc"]) < 1e-12
    assert "| leave one session out | **0.054** |" in README
    assert round(SUMM["pooled_auc"], 3) == 0.054


def test_session_table_rows():
    for s, r in SESS.items():
        row = "| %s | %s | %d | %s | %s |" % (
            NAME[s], "queenright" if r["queen"] else "queenless", r["n"],
            pct(r["called_queenright"]), prob(r["median_p"]))
        assert row in README, row


def test_no_session_is_called_right():
    assert SUMM["sessions_majority_correct"] == 0
    assert "No session is called right by a majority of its clips" in README
    assert SUMM["sessions_called_wrong_over_80pct"] == 7
    assert "Seven of the nine are called wrong on more than 80 %" in README
    assert SESS["Hive1_31_05_2018"]["n_called_queenright"] == 301
    assert "301 of 600" in README
    wrong_gh = 1 - SESS["GH001"]["called_queenright"]
    assert "%.0f %%" % (100 * wrong_gh) == "79 %" and "GH001 on 79 %" in README
    # the four single-session hives: 5 of 6, 87 %, 81 %, 79 %
    assert round(6 * SESS["CF001"]["called_queenright"]) == 5
    assert "%.0f" % (100 * (1 - SESS["CF003"]["called_queenright"])) == "87"
    assert "(5 of 6 clips, 87 %, 81 %, 79 %)" in README


def test_session_level_reversal_is_exact():
    assert SUMM["session_auc_of_mean_p"] == 0.0
    assert SUMM["relabellings"] == 126
    assert abs(SUMM["session_auc_exact_p_two_sided"] - 2 / 126) < 1e-12
    assert "Only 2 of the 126 relabellings" in README and "p = 0.016" in README


def test_training_prior_and_the_no_information_model():
    qr = SUMM["train_prior_when_queenright_held_out"]
    ql = SUMM["train_prior_when_queenless_held_out"]
    assert "%.0f-%.0f %%" % (100 * qr[0], 100 * qr[-1]) == "41-45 %"
    assert "%.0f-%.0f %%" % (100 * ql[0], 100 * ql[-1]) == "49-57 %"
    assert "41-45 % queenright" in README and "49-57 %" in README
    assert max(qr) < min(ql)
    assert SUMM["prior_only_pooled_auc"] == 0.0
    assert "pooled AUC 0.00, not 0.50" in README
    med = [r["median_p"] for r in SESS.values()]
    assert prob(min(med)) == "3e-07" and prob(max(med)) == "1.00"
    assert "medians from 3e-07 to 1.00" in README


def test_hive_protocol_gives_one_label_per_hive():
    h1, h3 = HIVE["Hive1"], HIVE["Hive3"]
    assert "%.1f" % (100 * (1 - h1["classes"]["0"]["called_queenright"])) == "99.8"
    assert h1["classes"]["1"]["called_queenright"] == 0.0
    assert "%.2f" % h1["within_hive_auc"] == "0.49"
    assert "%.0f" % (100 * h3["classes"]["0"]["called_queenright"]) == "75"
    assert "%.0f" % (100 * h3["classes"]["1"]["called_queenright"]) == "82"
    assert "%.2f" % h3["within_hive_auc"] == "0.38"
    assert "%.0f" % (100 * (1 - HIVE["CF003"]["called_queenright_all"])) == "76"
    assert "%.0f" % (100 * (1 - HIVE["GH001"]["called_queenright_all"])) == "79"
    assert HIVE["CF001"]["called_queenright_all"] == 1.0
    assert "%.0f" % (100 * (1 - HIVE["CJ001"]["called_queenright_all"])) == "65"
    for text in ("queenless on 99.8 % of its queenless clips and on 100 % of its queenright ones (within-hive AUC 0.49)",
                 "queenright on 75 % and 82 % (0.38)",
                 "called wrong on 76 % and 79 %, CF001 on 6 of 6",
                 "called right, on 65 %"):
        assert text in README, text


def test_wingbeat_table_rows():
    for s, r in WING.items():
        comb = "%.3f" % r["comb_share"] if r["comb_share"] < 0.01 else "%.2f" % r["comb_share"]
        row = "| %s | %s | %.2f | %.2f | %.2f | %.2f | %s |" % (
            NAME[s], META[s]["source"], r["mass_80_95"], r["mass_100_160"],
            r["mass_180_260"], r["p_at_100"], comb)
        assert row in README, row


def test_where_the_mass_sits():
    nu = [r["mass_100_160"] for s, r in WING.items() if META[s]["source"] == "NU-Hive"]
    os_ = [r["mass_100_160"] for s, r in WING.items() if META[s]["source"] == "OSBH"]
    assert len(nu) == 5 and len(os_) == 4
    assert "%.0f-%.0f %%" % (100 * min(nu), 100 * max(nu)) == "55-89 %"
    assert "%.0f-%.0f %%" % (100 * min(os_), 100 * max(os_)) == "7-42 %"
    assert min(nu) > 0.5 > max(os_)
    edge = [r["mass_80_95"] for s, r in WING.items() if META[s]["source"] == "OSBH"]
    assert "%.0f-%.0f %%" % (100 * min(edge), 100 * max(edge)) == "21-84 %"
    assert "%.0f" % (100 * WING["CJ001"]["p_at_80"]) == "30"
    assert "%.0f" % (100 * WING["GH001"]["p_at_80"]) == "31"
    band = sorted(r["mass_180_260"] for r in WING.values())
    assert all(b < 0.035 for b in band[:5])
    assert ["%.0f" % (100 * b) for b in band[5:]] == ["13", "21", "26", "30"]
    for text in ("sessions (55-89 %) and of none of the four citizen-science hives (7-42 %)",
                 "hold 21-84 % of the mass", "holds 30 % in CJ001 and 31 % in GH001",
                 "it holds 13 %, 21 %, 26 % and 30 % in the other four"):
        assert text in README, text
    assert "%.1f" % (100 * WING["CF001"]["comb_share"]) == "0.3"
    assert "assigns 0.3 % of the power to wing beats" in README


def test_the_100_hz_line_is_in_two_sessions():
    a, b = WING["Hive3_15_07_2017"], WING["Hive3_20_07_2017"]
    assert "%.0f %.0f" % (100 * a["p_at_100"], 100 * b["p_at_100"]) == "15 21"
    assert "%.1f %.1f" % (a["line_ratio_100"], b["line_ratio_100"]) == "6.1 5.2"
    rest = [r for s, r in WING.items() if s not in ("Hive3_15_07_2017", "Hive3_20_07_2017")]
    assert len(rest) == 7
    assert max(r["p_at_100"] for r in rest) < 0.05
    ratios = sorted(r["line_ratio_100"] for r in rest)
    assert "%.1f-%.1f" % (ratios[0], ratios[-1]) == "0.2-1.6"
    assert "%.1f" % (100 * WING["Hive3_12_07_2017"]["p_at_100"]) == "0.4"
    for text in ("15 % and 21 % of the mass in one grid point, 6.1 and 5.2 times",
                 "0-5 % and 0.2-1.6 times its neighbours", "has 0.4 % there"):
        assert text in README, text


def test_named_quantities_rows():
    label = {"f0_mean": ("mean f0 (Hz)", "%.0f"), "f0_sd": ("spread of f0 (Hz)", "%.0f"),
             "comb_share": ("comb share", "%.2f"), "bg_slope": ("background slope", "%.2f")}
    for k, r in NAMED.items():
        name, fmt = label[k]
        row = "| %s | %s | %s | %.2f | %.2f | %.2f |" % (
            name, fmt % r["queenright_mean"], fmt % r["queenless_mean"],
            r["session_auc"], r["exact_p"], r["exact_p_without_CF001"])
        assert row in README, row
        assert r["exact_p"] > 0.35 and r["exact_p_without_CF001"] > 0.5


def test_clip_counts_differ_between_rows():
    assert CHK["n_clips"] == {"cnn": 4301, "wingbeat_and_band_rows": 2406}
    assert WB["n_clips"] == 2406 and SUMM["n_clips"] == 4301
    assert "use 4 301 clips" in README and "use 2 406" in README


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok  ", fn.__name__)
    print(f"{len(fns)} checks passed")
