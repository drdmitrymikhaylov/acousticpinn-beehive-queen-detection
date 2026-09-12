# Is the queen there? What hive sound can and cannot say

A queenless colony is a colony with weeks to live, and beekeepers have long
said they can hear it. Published classifiers agree, with accuracies in the
high nineties. This repository asks the only question that matters to someone
who wants to put a microphone in *their* hive: does any of that survive a
colony the model has never heard?

Three readings of the same recordings are put through the same three tests.
A log-mel convolutional network (what the literature does), eight band
energies (what a beekeeper's chart shows), and a **physical model of the hum
as a population of wing beats**, which can only explain a spectrum through
the harmonics of a wing stroke and returns per-clip numbers with names — the
colony's wing-beat frequency, its spread, the share of the sound that is
bees at all.

---

## The data

Inês Nolasco and Emmanouil Benetos, *To bee or not to bee: an annotated
dataset for beehive sound recognition* (Zenodo 1321278, CC BY 4.0). Two
sources, and the difference between them is the whole design:

| source | hives | sessions | what a label is |
|---|---|---|---|
| NU-Hive | 2 (Hive1, Hive3) | 5 | the same hive recorded on separate days with the queen present and after she was removed |
| Open Source Beehives | 4 citizen-science hives | 4 | one hive, one state: *Active* or *Missing Queen* |

Every file carries an annotation of which seconds are hive sound and which
are a passing car or a voice; only the hive-sound seconds are used, cut into
two-second clips at 8 kHz. 56 files, 10 277 clips, 5.7 hours; 4 301 clips
after capping each session at 600 so that no single afternoon dominates.

Nine sessions, six hives. That is the true sample size for any claim about
queens, and every number below is read with it in mind.

---

## What came out

| # | Finding | Evidence |
|---|---------|----------|
| 1 | Two thirds to four fifths of a hive's acoustic energy is below 260 Hz, and the sessions of one class differ from each other by more than the classes differ. The textbook 180–260 Hz "worker flight" band holds 7 % of the power in queenright and 6 % in queenless sessions. | §1 |
| 2 | **A random split over clips gives AUC 0.9999.** The same network, asked about a session it has not heard, gives **0.05**; about a hive it has not heard, **0.38**. Below chance is not noise: the network learned to recognise sessions, and the nearest session usually has the other label. | §2 |
| 3 | The hum is well described as a harmonic comb — a learned single-bee spectrum with a 2.4 Hz line width that widens with harmonic number — but the model places most of the colony's fundamental at 100–160 Hz, not at the textbook 200–250 Hz. Either these recordings hum lower than the literature says, or the model is reading something periodic that is not a bee (see below). | §3 |
| 4 | Five named numbers from the wing-beat model transfer where the network does not: AUC **0.60** across sessions and **0.62** across hives, against 0.52 / 0.57 for band energies. That is above chance and nowhere near a product. | §3 |
| 5 | The full wing-beat distribution p(f0), fed to the same regression, does worse than chance across sessions (0.20): fifty-five numbers per clip are enough to learn session identity again. The physics helps only when it is reduced to the few quantities that mean something. | §3 |

---

## 1. What a hive sounds like

![session spectra](figures/01_session_spectra.png)

One curve per recording session, area-normalised, blue queenright and orange
queenless. Two things are visible before any model is fitted. The energy is
low: most of it sits under 260 Hz, and the shape of the spectrum differs
between the two projects (different microphones, boxes and countries) more
than between the two queen states. And within a project, the two queenless
sessions of Hive3 do not look more like each other than either looks like the
queenright session.

![band shares](figures/02_band_shares.png)

The bands the literature names, one line per session. The unit of replication
is the session, not the clip: ten thousand two-second clips from one
afternoon in one hive are one observation of one colony, and treating them
as independent is how a difference between two recordings becomes a p-value
with ten zeros in it. Read that way, no band separates the classes by more
than the spread within a class. The 500–600 Hz "swarming band" holds 2.7 %
versus 1.3 % — a difference, with a standard deviation across sessions of
1.5 and 1.0.

Because there are only nine sessions, "larger than the spread" can be made
exact. Four queenright labels among nine sessions can be handed out in
C(9, 4) = 126 ways, so every relabelling is enumerated and the observed gap
between class means is placed among all 126 (`src/band_permutation.py`,
`results/band_permutation.json`). The smallest two-sided p-value nine
sessions can produce is 1/126 = 0.008; no band comes close:

| band | queenright | queenless | session AUC | exact p | p without CF001 |
|---|---|---|---|---|---|
| 20–100 Hz | 0.33 | 0.43 | 0.45 | 0.61 | 0.94 |
| 100–180 Hz | 0.26 | 0.30 | 0.45 | 0.71 | 0.57 |
| 180–260 Hz (worker flight) | 0.069 | 0.061 | 0.65 | 0.76 | 0.91 |
| 260–400 Hz | 0.136 | 0.078 | 0.75 | 0.32 | 0.51 |
| 400–500 Hz | 0.044 | 0.034 | 0.65 | 0.56 | 0.91 |
| 500–600 Hz (swarming) | 0.027 | 0.013 | 0.75 | 0.18 | 0.34 |
| 600–1000 Hz | 0.061 | 0.036 | 0.70 | 0.44 | 0.63 |
| 1–4 kHz | 0.039 | 0.029 | 0.55 | 0.68 | 1.00 |

Shares of session power; session AUC is the fraction of the twenty
queenright/queenless session pairs in which the queenright session has the
larger share. The last column drops CF001, a queenless session that
contributes six clips yet carries the same weight as a 1 200-clip session in
every session-level mean on this page; the 20–100 Hz gap, the largest in the
table, is mostly that one session. The swarming band, the best of the eight,
is at p = 0.18 with all sessions and 0.34 without it — the same finding as
above, now with a number on it.

---

## 2. The same model, three ways of splitting the data

![protocols](figures/03_protocols.png)

A small convolutional network on log-mel spectrograms (filterbank from 20 Hz,
so the low band is not discarded by a default), trained the same way three
times:

- **random** — five stratified folds over clips. Clips from the same
  ten-minute file land on both sides. This is how most published results are
  produced.
- **session** — leave one recording session out.
- **hive** — leave one hive out. The only protocol that answers the question
  a beekeeper asks.

Folds under the last two often contain a single class, so out-of-fold scores
are pooled and scored once.

| protocol | pooled AUC | balanced accuracy |
|---|---|---|
| random over clips | **0.9999** | 0.995 |
| leave one session out | **0.054** | 0.14 |
| leave one hive out | **0.375** | 0.39 |

The random number is real and means nothing: it measures whether the network
can tell which recording a clip came from. The session number is the
instructive one. An AUC of 0.05 is not a model that knows nothing; it is a
model that confidently assigns each new session the label of the session it
most resembles — and in this dataset the nearest session usually has the
other label (Hive1's two days, Hive3's three). A model that had learned
*queens* would sit near 0.5 on a held-out session at worst. This one learned
*sessions*, and it says so.

---

## 3. Putting the physics in: the hum as a population of wing beats

![wing-beat model](figures/04_wingbeat_model.png)

A hive sounds the way it does because tens of thousands of bees beat their
wings. Each is a periodic source — a stroke at a fundamental f₀ that moves
with the bee's size, load and temperature — and a periodic source has
harmonics. The colony spectrum is written as that superposition and nothing
else:

    S(f) = g · [ Σₖ p(f₀ₖ) Σₕ a(h, f₀ₖ) L(f; h·f₀ₖ, h·γ)  +  B(f) ]

with p(f₀) the distribution of wing-beat frequencies in the colony (a
55-point histogram from 80 to 350 Hz, one per clip), a(h, f₀) the harmonic
profile of a stroke — a small network of harmonic number and fundamental,
shared by every clip — a Lorentzian line whose width grows with harmonic
number, and B(f) a power-law background per clip for everything that is not a
wing beat. The model is fitted to the spectra alone, by log-spectral error;
**the queen label never enters the fit.** Under the session and hive
protocols the shared physics (line width, harmonic profile) is refitted on
the training sessions only and frozen before the held-out clips are
decomposed.

**What the fit finds.** A single-bee spectrum with a line width of 2.4 Hz at
the fundamental, and a harmonic profile that decays over five harmonics. The
learned colony distributions put most of the mass at 100–160 Hz, with the
textbook 180–260 Hz band nearly empty in every session. Where the mass sits
differs by session far more than by class: the queenless sessions include one
at 104 Hz and, on six clips, one at 245 Hz.

**What the numbers are worth.** Five quantities per clip — mean and spread of
f₀, share of power in the comb, background slope and level — go into a
logistic regression under the same three protocols, next to the eight band
energies and the network:

| reading of the sound | random | session | hive |
|---|---|---|---|
| log-mel CNN | 0.9999 | 0.054 | 0.375 |
| 8 band energies + logistic regression | 0.915 | 0.522 | 0.566 |
| wing-beat model, full p(f₀) (55 numbers) | 0.909 | 0.203 | 0.387 |
| **wing-beat model, 5 named numbers** | 0.842 | **0.595** | **0.618** |

Read across a row and the pattern is the same one this repository keeps
finding: the more freedom a reading has, the better it does on the random
split and the worse it does on a new colony. The network is best on clips it
has effectively seen and worst on colonies it has not. The full wing-beat
histogram, 55 numbers per clip, is enough to identify sessions again and
falls to 0.20. Only the five quantities with physical names stay above chance
on a new session (0.60) and a new hive (0.62) — modestly, on nine sessions,
and not enough to act on.

**The caveat that matters.** The model knows what a harmonic series looks
like; it does not know what a bee is. Any periodic source in the recording
— mains hum at 50 or 60 Hz and its harmonics, a fan, a compressor in the
citizen-science recordings — is decomposed into "bees" at whatever
fundamental fits. The mass at 100 and 150 Hz in several sessions is exactly
where the harmonics of a 50 Hz mains would sit. This is the same lesson as
§2 in a different coat: the decomposition is only as honest as the
recording, and a physical model makes that failure *legible* — a
distribution parked at mains harmonics is visible in the left panel — where
a spectrogram network makes it invisible.

---

## Verification

Eight checks, all passing:

- the queen label, hive and session are parsed from every file-name
  pattern in the dataset, and documentation files are rejected
- only segments annotated as hive sound are kept; a comma decimal is read
  correctly
- the mel filterbank covers every frequency from 40 Hz up, so the band
  where hive sound lives is not silently discarded
- a single bee in the wing-beat model produces peaks at f₀, 2f₀, 3f₀, 4f₀
  and nowhere else
- the line width at the third harmonic is three times the width at the
  first
- a colony placed entirely at 250 Hz is read back as f₀ = 250 Hz with zero
  spread
- **the result is pinned**: the random-split AUC must exceed 0.99, the
  network's session and hive AUCs must be below 0.5, and the five wing-beat
  numbers must land between 0.5 and 0.8 on both honest protocols — the page
  is not allowed to claim a working detector
- **no band separates the classes**: the exact permutation test over the
  126 relabellings of nine sessions returns p > 0.1 for every band, with
  and without the six-clip session, and the estimator returns 1/126 on a
  perfectly separated case

---

## What this does not show

- **Nine sessions from six hives.** Every cross-colony number above has a
  sample size of six, not four thousand. The 0.60 / 0.62 could be 0.5 on the
  next six hives.
- **Two recording projects, two microphones, two countries.** Part of what
  separates sessions is equipment. A dataset with one microphone in twenty
  hives would answer the hive question far better than this one can.
- **Queen removal is not the only thing that changed** between the NU-Hive
  sessions: date, weather, time of day and forage did too. The label is the
  queen; the sound is everything.
- **The wing-beat model is a spectral model, not a bee counter.** It cannot
  tell a bee from any other periodic source, and it does not model the
  box, which shapes the spectrum below 300 Hz.
- **No swarming, no disease, no robbing.** The dataset labels one state.

---

## Source code

The physics core is public in this repository:

- `src/prepare.py` — the recordings to labelled clips, with provenance
- `src/spectra.py` — session-level spectra and the band table
- `src/model.py` — the log-mel front end (filterbank written out) and the
  network
- `src/train.py` — the three protocols, pooled scoring, resumable
- `src/band_permutation.py` — the exact permutation test on the session
  band table (standard library only)
- `src/wingbeat_pinn.py` — the wing-beat mixture model, its named features,
  and the protocol comparison
- `tests/test_all.py` — the eight checks above

`results/` holds every number on this page as JSON. `data/SOURCE.md` says
how to fetch the recordings; they are not redistributed here.

```
pip install -r requirements.txt
python src/download.py && python src/prepare.py
python src/spectra.py && python src/band_permutation.py
python src/train.py                               # ~25 min on Apple silicon
python src/wingbeat_pinn.py                       # ~20 min on a laptop CPU
python src/figures.py && python tests/test_all.py
```

---

## Licence and credit

Documentation, figures and result files: CC BY 4.0. Source code in `src/`
and `tests/`: MIT.

The recordings belong to their authors and are distributed through Zenodo
under CC BY 4.0. Cite them, not this page:

- Inês Nolasco, Emmanouil Benetos. 2018. *To bee or not to bee: An annotated
  dataset for beehive sound recognition.* Zenodo, doi:10.5281/zenodo.1321278.
  Recordings from the NU-Hive project and the Open Source Beehives project.

---

*One of a series of physics-informed acoustic projects; see the profile
[README](https://github.com/drdmitrymikhaylov) for the others.*
