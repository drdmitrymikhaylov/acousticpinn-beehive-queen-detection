"""The hum as a population of wing beats, not as a spectrogram.

A hive sounds the way it does because tens of thousands of bees beat their
wings.  Each bee is a periodic source: a wing stroke at a fundamental f0
(around 200-250 Hz for a worker; it moves with the bee's size, load and
temperature) and the harmonics of that stroke.  The colony spectrum is the
superposition,

    S(f) = g * [ sum_k p(f0_k) sum_h a(h, f0_k) L(f; h f0_k, h gamma)  +  B(f) ],

with p(f0) the distribution of wing-beat frequencies in the colony, a(h, f0)
the harmonic profile of a stroke (a small network of harmonic number and
fundamental), L a Lorentzian line of width proportional to the harmonic
number, and B(f) a smooth power-law background for everything that is not a
wing beat (ventilation, the box, the microphone).

The classifier in train.py sees a log-mel image and is free to learn whatever
tells the sessions apart.  This model can only explain a spectrum through
wing beats and a background, and the numbers it returns per clip have names:
the mean and spread of the colony's wing-beat frequency, the share of power
in the harmonic comb, the background slope.  Whether those numbers carry the
queen's status -- and whether they carry it *across hives*, where the
spectrogram model fails -- is the question this file asks.

Everything is fitted to the spectra alone; the queen label is used only
afterwards, in a logistic regression evaluated under the same three protocols
as train.py.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from train import folds_for, load_all, SR                    # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
torch.set_default_dtype(torch.float64)

F_LO, F_HI = 20.0, 1500.0
F0_GRID = np.arange(80.0, 351.0, 5.0)       # wing-beat fundamentals, Hz; the grid starts
                                            # well below the textbook 200-250 Hz so the
                                            # data, not the literature, place the hum
N_HARM = 6
MAX_CLIPS_PER_SESSION = 300
BANDS = [(20, 100), (100, 180), (180, 260), (260, 400), (400, 500),
         (500, 600), (600, 1000), (1000, 1500)]


def spectra(X, sr=SR, nperseg=4096):
    f, P = welch(X, fs=sr, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
    m = (f >= F_LO) & (f <= F_HI)
    return f[m], P[:, m]


def mlp(n_in, width=32):
    return torch.nn.Sequential(torch.nn.Linear(n_in, width), torch.nn.Tanh(),
                               torch.nn.Linear(width, width), torch.nn.Tanh(),
                               torch.nn.Linear(width, 1))


class WingbeatMixture(torch.nn.Module):
    def __init__(self, n_clips, f, f0_grid=F0_GRID, n_harm=N_HARM):
        super().__init__()
        self.register_buffer("f", torch.tensor(f))
        self.register_buffer("f0", torch.tensor(f0_grid))
        self.register_buffer("h", torch.arange(1, n_harm + 1, dtype=torch.float64))
        self.p_logit = torch.nn.Parameter(torch.zeros(n_clips, len(f0_grid)))
        self.log_gamma = torch.nn.Parameter(torch.tensor(np.log(4.0)))     # Hz at h = 1
        self.profile = mlp(2)                                                # a(h, f0)
        self.bg = torch.nn.Parameter(torch.tensor([[-2.0, 1.0, -4.0]] * n_clips))
        self.log_gain = torch.nn.Parameter(torch.zeros(n_clips, 1))

    def harmonic_profile(self):
        H, K = len(self.h), len(self.f0)
        hh = self.h.reshape(H, 1).expand(H, K)
        ff = self.f0.reshape(1, K).expand(H, K)
        inp = torch.stack([(hh - 3.5) / 2.0, (ff - 215.0) / 80.0], -1)
        return torch.nn.functional.softplus(self.profile(inp).squeeze(-1))     # (H, K)

    def comb_basis(self):
        """(K, F): spectrum of one bee at fundamental f0_k, harmonics summed."""
        gamma = torch.exp(self.log_gamma) * self.h                              # (H,)
        centre = self.h.reshape(-1, 1) * self.f0.reshape(1, -1)                 # (H, K)
        d = self.f.reshape(1, 1, -1) - centre.unsqueeze(-1)                     # (H, K, F)
        g = gamma.reshape(-1, 1, 1)
        L = (g / np.pi) / (d ** 2 + g ** 2)
        a = self.harmonic_profile().unsqueeze(-1)                               # (H, K, 1)
        return (a * L).sum(0)                                                   # (K, F)

    def parts(self):
        p = torch.softmax(self.p_logit, 1)                                      # (N, K)
        comb = p @ self.comb_basis()                                            # (N, F)
        lf = torch.log(self.f / 100.0).reshape(1, -1)
        b0, b1, c = self.bg[:, :1], self.bg[:, 1:2], self.bg[:, 2:]
        bg = torch.exp(b0 - torch.nn.functional.softplus(b1) * lf) + torch.exp(c)
        return p, comb, bg

    def forward(self):
        p, comb, bg = self.parts()
        return torch.exp(self.log_gain) * (comb + bg)


def fit(P, f, steps=1500, lr=2e-2, model=None, freeze_shared=False, log=print):
    Y = torch.tensor(P)
    Y = Y / Y.mean(1, keepdim=True)
    N = Y.shape[0]
    m = WingbeatMixture(N, f)
    if model is not None:
        m.log_gamma.data.copy_(model.log_gamma.data)
        m.profile.load_state_dict(model.profile.state_dict())
    params = [m.p_logit, m.bg, m.log_gain]
    if freeze_shared:
        m.log_gamma.requires_grad_(False)
        for q in m.profile.parameters():
            q.requires_grad_(False)
    else:
        params += [m.log_gamma] + list(m.profile.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    logY = torch.log(Y + 1e-9)
    for it in range(steps):
        opt.zero_grad()
        loss = torch.mean((torch.log(m() + 1e-9) - logY) ** 2)
        loss.backward()
        opt.step()
        if it % 300 == 0 or it == steps - 1:
            log(f"    step {it:5d}  log-spectral loss {float(loss):.4f}  "
                f"gamma {float(torch.exp(m.log_gamma)):.2f} Hz")
    return m, float(loss)


def features(m):
    """Named per-clip quantities from the fitted mixture."""
    with torch.no_grad():
        p, comb, bg = m.parts()
        f0 = m.f0
        mean = (p * f0).sum(1)
        sd = torch.sqrt((p * (f0 - mean[:, None]) ** 2).sum(1))
        comb_share = comb.sum(1) / (comb.sum(1) + bg.sum(1))
        slope = torch.nn.functional.softplus(m.bg[:, 1])
        level = m.bg[:, 0] - m.bg[:, 2]
        F = torch.stack([mean, sd, comb_share, slope, level], 1).numpy()
    return F, p.numpy()


def band_features(P, f):
    out = []
    for lo, hi in BANDS:
        k = (f >= lo) & (f < hi)
        out.append(np.log(P[:, k].mean(1) + 1e-12))
    return np.stack(out, 1)


def evaluate(Fdict, y, sess, hive, log=print):
    """Logistic regression on each feature set, pooled AUC under 3 protocols."""
    out = {}
    for name, F in Fdict.items():
        out[name] = {}
        for protocol in ("random", "session", "hive"):
            yy, pp = [], []
            for _, tr, te in folds_for(protocol, y, sess, hive):
                sc = StandardScaler().fit(F[tr])
                clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)
                clf.fit(sc.transform(F[tr]), y[tr])
                pp.append(clf.predict_proba(sc.transform(F[te]))[:, 1]); yy.append(y[te])
            yy, pp = np.concatenate(yy), np.concatenate(pp)
            out[name][protocol] = {"pooled_auc": float(roc_auc_score(yy, pp)), "n": int(yy.size)}
        log(f"  {name:22s} " + "  ".join(f"{k} {v['pooled_auc']:.3f}" for k, v in out[name].items()))
    return out


def main():
    RESULTS.mkdir(exist_ok=True)
    import train
    train.MAX_CLIPS_PER_SESSION = MAX_CLIPS_PER_SESSION
    X, y, sess, hive = load_all()
    f, P = spectra(X)
    print(f"{len(y)} clips, {len(set(sess))} sessions, {len(set(hive))} hives, "
          f"{len(f)} frequency bins {f[0]:.0f}-{f[-1]:.0f} Hz", flush=True)

    # one fit on everything: the shared physics (line width, harmonic profile)
    print("global fit", flush=True)
    m_all, loss_all = fit(P, f, log=lambda s: print(s, flush=True))
    F_all, p_all = features(m_all)
    with torch.no_grad():
        basis = m_all.comb_basis().numpy()
        prof = m_all.harmonic_profile().numpy()
        fit_spec = m_all().numpy()
    Yn = P / P.mean(1, keepdims=True)
    rel = float(np.sqrt(np.mean((np.log(fit_spec + 1e-9) - np.log(Yn + 1e-9)) ** 2)))

    # session-level summary of the physical quantities
    sessions = sorted(set(sess))
    per_session = {}
    for s in sessions:
        k = sess == s
        per_session[s] = {"queen": int(y[k][0]), "hive": str(hive[k][0]), "n": int(k.sum()),
                          "f0_mean": float(F_all[k, 0].mean()), "f0_sd": float(F_all[k, 1].mean()),
                          "comb_share": float(F_all[k, 2].mean()),
                          "bg_slope": float(F_all[k, 3].mean()),
                          "p_f0": p_all[k].mean(0).tolist()}
        print(f"  {s:24s} queen={per_session[s]['queen']}  f0 {per_session[s]['f0_mean']:.0f} "
              f"+/- {per_session[s]['f0_sd']:.0f} Hz  comb share {per_session[s]['comb_share']:.2f}"
              f"  bg slope {per_session[s]['bg_slope']:.2f}", flush=True)

    # the honest protocol test: shared physics from the training sessions only,
    # per-clip quantities on the held-out ones with the physics frozen
    print("protocols (physics refitted per fold, labels only in the regression)", flush=True)
    F_band = band_features(P, f)
    res = {"random": {}, "session": {}, "hive": {}}
    for protocol in ("session", "hive", "random"):
        folds = folds_for(protocol, y, sess, hive)
        yy, pp_phys, pp_p, pp_band = [], [], [], []
        for name, tr, te in folds:
            m_tr, _ = fit(P[tr], f, steps=900, log=lambda s: None)
            m_te, _ = fit(P[te], f, steps=500, model=m_tr, freeze_shared=True, log=lambda s: None)
            Ftr, ptr = features(m_tr); Fte, pte = features(m_te)
            for F_train, F_test, store in ((Ftr, Fte, pp_phys), (ptr, pte, pp_p),
                                           (F_band[tr], F_band[te], pp_band)):
                sc = StandardScaler().fit(F_train)
                clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)
                clf.fit(sc.transform(F_train), y[tr])
                store.append(clf.predict_proba(sc.transform(F_test))[:, 1])
            yy.append(y[te])
            print(f"  {protocol:8s} {str(name):22s} n_test={len(te):5d}", flush=True)
        yy = np.concatenate(yy)
        for key, store in (("wingbeat_5_features", pp_phys), ("wingbeat_distribution", pp_p),
                           ("band_energies", pp_band)):
            res[protocol][key] = {"pooled_auc": float(roc_auc_score(yy, np.concatenate(store))),
                                  "n": int(yy.size)}
        print(f"  {protocol}: " + "  ".join(f"{k} {v['pooled_auc']:.3f}" for k, v in res[protocol].items()),
              flush=True)

    cnn = {}
    ps = ROOT / "results" / "protocol_summary.json"
    if ps.exists():
        cnn = {k: v["pooled_auc"] for k, v in json.loads(ps.read_text()).items()}
    out = {"n_clips": int(len(y)), "f": f.tolist(), "f0_grid": F0_GRID.tolist(),
           "gamma_hz": float(torch.exp(m_all.log_gamma)), "harmonic_profile": prof.tolist(),
           "comb_basis_example": basis[len(F0_GRID) // 2].tolist(),
           "global_log_rmse": rel, "per_session": per_session, "protocols": res,
           "cnn_pooled_auc": cnn}
    (RESULTS / "wingbeat_pinn.json").write_text(json.dumps(out, indent=1))
    print("saved", flush=True)


if __name__ == "__main__":
    main()
