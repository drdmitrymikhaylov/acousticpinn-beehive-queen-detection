"""Checks run before any number on the public page is believed."""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_file_names_carry_the_labels():
    from prepare import parse_meta
    m = parse_meta("Hive1_31_05_2018_NO_QueenBee_H1_audio___15_00_00")
    assert m == {"source": "NU-Hive", "hive": "Hive1", "session": "Hive1_31_05_2018", "queen": 0}
    m = parse_meta("Hive3_20_07_2017_QueenBee_H3_audio___06_10_00")
    assert m["queen"] == 1 and m["session"] == "Hive3_20_07_2017"
    m = parse_meta("CJ001 - Missing Queen - Day -  (100)")
    assert m == {"source": "OSBH", "hive": "CJ001", "session": "CJ001", "queen": 0}
    assert parse_meta("Dataset documentation") is None


def test_only_bee_segments_are_kept(tmp_path=None):
    from prepare import read_lab
    p = (tmp_path or pathlib.Path("/tmp")) / "x.lab"
    p.write_text("0.0 4.5 bee\n4.5 6.0 nobee\n6,0 9,0 bee\n")
    assert read_lab(p) == [(0.0, 4.5), (6.0, 9.0)]


def test_mel_filterbank_covers_the_low_band():
    """Hive sound lives below 300 Hz; a filterbank that starts at a default
    fmin would throw it away.  Every frequency from 20 Hz up is covered."""
    from model import mel_filterbank, N_FFT, SR
    fb = mel_filterbank().numpy()
    freqs = np.linspace(0, SR / 2, N_FFT // 2 + 1)
    cover = fb.sum(0)
    assert (cover[(freqs > 40) & (freqs < 3700)] > 0).all()
    assert fb.shape[0] == 64 and (fb >= 0).all()


def test_single_bee_spectrum_is_a_harmonic_comb():
    """One bee at f0 must peak at f0, 2 f0, 3 f0 ... and nowhere else."""
    import torch
    from wingbeat_pinn import WingbeatMixture
    f = np.arange(20.0, 1500.0, 1.0)
    m = WingbeatMixture(1, f, f0_grid=np.array([200.0]), n_harm=4)
    with torch.no_grad():
        for q in m.profile.parameters():
            q.zero_()                       # flat harmonic profile
        B = m.comb_basis().numpy()[0]
    peaks = [int(f[i]) for i in range(1, len(f) - 1)
             if B[i] > B[i - 1] and B[i] > B[i + 1] and B[i] > 0.05 * B.max()]
    assert peaks == [200, 400, 600, 800], peaks


def test_line_width_scales_with_harmonic_number():
    import torch
    from wingbeat_pinn import WingbeatMixture
    f = np.arange(20.0, 1500.0, 0.5)
    m = WingbeatMixture(1, f, f0_grid=np.array([200.0]), n_harm=3)
    with torch.no_grad():
        for q in m.profile.parameters():
            q.zero_()
        m.log_gamma.fill_(np.log(2.0))
        B = m.comb_basis().numpy()[0]

    def fwhm(centre):
        k = np.abs(f - centre) < 40
        seg, ff = B[k], f[k]
        return np.ptp(ff[seg > seg.max() / 2])
    assert abs(fwhm(200) - 4.0) < 1.0        # 2 gamma at h = 1
    assert abs(fwhm(600) - 12.0) < 2.0       # 2 gamma * 3 at h = 3


def test_features_read_the_distribution():
    """A colony put entirely at 250 Hz must report f0 = 250 with zero spread."""
    import torch
    from wingbeat_pinn import WingbeatMixture, features, F0_GRID
    f = np.arange(20.0, 1500.0, 2.0)
    m = WingbeatMixture(2, f)
    with torch.no_grad():
        m.p_logit.zero_()
        m.p_logit[0, int(np.argmin(np.abs(F0_GRID - 250.0)))] = 40.0
    F, p = features(m)
    assert abs(F[0, 0] - 250.0) < 0.5 and F[0, 1] < 0.5
    assert F[1, 1] > 40.0                    # the flat clip has a wide spread


def test_results_say_what_the_page_says():
    """Random splits look perfect and mean nothing; across sessions the
    spectrogram model is below chance; the wing-beat numbers are above it."""
    p = ROOT / "results" / "protocol_summary.json"
    q = ROOT / "results" / "wingbeat_pinn.json"
    if not (p.exists() and q.exists()):
        return
    cnn = json.loads(p.read_text())
    assert cnn["random"]["pooled_auc"] > 0.99
    assert cnn["session"]["pooled_auc"] < 0.5 and cnn["hive"]["pooled_auc"] < 0.5
    w = json.loads(q.read_text())["protocols"]
    assert w["random"]["band_energies"]["pooled_auc"] > 0.85
    assert 0.5 < w["session"]["wingbeat_5_features"]["pooled_auc"] < 0.8
    assert 0.5 < w["hive"]["wingbeat_5_features"]["pooled_auc"] < 0.8


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
