# Is the queen there? What hive sound can and cannot say

A queenless colony has weeks to live, and beekeepers have long said they
can hear it. Published classifiers agree, with accuracies in the high
nineties. This repository asks a narrower question. Someone who wants to put
a microphone in *their* hive needs to know whether any of that survives a
colony the model has never heard.

Three readings of the same recordings go through the same three tests. The
first is a log-mel convolutional network, which is what the literature uses.
The second is eight band energies, which is what a beekeeper's chart shows.
The third is a **physical model of the hum built from wing beats**. That model
can only explain a spectrum through the harmonics of a wing stroke. It
returns per-clip quantities that have a physical meaning: the colony's
wing-beat frequency, its spread, and the share of the sound that is bees at
all.

---

## Where this comes from

Hive acoustics came to me from the field: beekeepers and agricultural clients who lose colonies because nobody notices a queenless hive until it is too late. A hive tells you everything through its hum, if you know which physics to listen for. I have worked on that signal for years in research and client projects, and this repository puts the core of it in the open: a wing-beat model of the colony built from first principles and tested on public recordings from six hives.

## 1. The recordings

Inês Nolasco and Emmanouil Benetos, *To bee or not to bee: an annotated
dataset for beehive sound recognition* (Zenodo 1321278, CC BY 4.0). There
are two sources, and the difference between them is the whole design:

| source | hives | sessions | what a label is |
|---|---|---|---|
| NU-Hive | 2 (Hive1, Hive3) | 5 | the same hive recorded on separate days with the queen present and after she was removed |
| Open Source Beehives | 4 citizen-science hives | 4 | one hive, one state: *Active* or *Missing Queen* |

Every file carries an annotation of which seconds are hive sound and which
are a passing car or a voice. Only the hive-sound seconds are used, cut into
two-second clips at 8 kHz. That gives 56 files, 10 277 clips and 5.7 hours.
After capping each session at 600 clips, so that no single afternoon
dominates, 4 301 clips remain.

Nine sessions, six hives. That is the true sample size for any claim about
queens, and every number below should be read with it in mind.

---

## 2. Findings at a glance

| # | Finding | Evidence |
|---|---------|----------|
| 1 | Two thirds to four fifths of a hive's acoustic energy is below 260 Hz, and the sessions of one class differ from each other by more than the classes differ. The textbook 180-260 Hz "worker flight" band holds 7 % of the power in queenright and 6 % in queenless sessions. | section 3 |
| 2 | **A random split over clips gives AUC 0.9999.** The same network, asked about a session it has not heard, gives **0.05**; about a hive it has not heard, **0.38**. Below chance is not noise. The network learned to recognise sessions: no held-out session is called right by a majority of its clips, and every queenless session scores above every queenright one (2 of 126 relabellings are that extreme, p = 0.016). The "nearest session has the other label" explanation given first covers five of the nine sessions at most. | sections 4, 4.1 |
| 3 | The hum is well described as a harmonic comb - a learned single-bee spectrum with a 2.4 Hz line width that widens with harmonic number. But the model places most of the colony's fundamental at 100-160 Hz, not at the textbook 200-250 Hz - in the five NU-Hive sessions. In the four citizen-science hives it piles the mass against the 80 Hz floor of its grid instead. Either these recordings hum lower than the literature says, or the model is reading something periodic that is not a bee (see below). | sections 5, 5.1 |
| 4 | Five named quantities from the wing-beat model transfer where the network does not: AUC **0.60** across sessions and **0.62** across hives, against 0.52 / 0.57 for band energies. That is above chance and nowhere near a product. | section 5 |
| 5 | The full wing-beat distribution p(f0), fed to the same regression, does worse than chance across sessions (0.20). Fifty-five numbers per clip are enough to learn session identity again. The physics helps only when it is reduced to the few quantities that mean something. | section 5 |

---

## 3. What a hive sounds like

![session spectra](figures/01_session_spectra.png)

One curve per recording session, area-normalised, blue queenright and orange
queenless. Two things are visible before any model is fitted. The energy is
low: most of it sits under 260 Hz. The shape of the spectrum differs between
the two projects (different microphones, boxes and countries) more than
between the two queen states. And within a project, the two queenless
sessions of Hive3 do not look more like each other than either looks like the
queenright session.

![band shares](figures/02_band_shares.png)

These are the bands the literature names, one line per session. The thing
being replicated is the session, not the clip. Ten thousand two-second clips
from one afternoon in one hive are one observation of one colony. Treating
them as independent is how a difference between two recordings becomes a
p-value with ten zeros in it. Read that way, no band separates the classes
by more than the spread within a class. The 500-600 Hz "swarming band" holds
2.7 % versus 1.3 % - a difference, with a standard deviation across sessions
of 1.5 and 1.0.

Because there are only nine sessions, "larger than the spread" can be made
exact. Four queenright labels among nine sessions can be handed out in
C(9, 4) = 126 ways. So every relabelling is enumerated and the observed gap
between class means is placed among all 126 (`src/band_permutation.py`,
`results/band_permutation.json`). The smallest two-sided p-value nine
sessions can produce is 1/126 = 0.008. No band comes close:

| band | queenright | queenless | session AUC | exact p | p without CF001 |
|---|---|---|---|---|---|
| 20-100 Hz | 0.33 | 0.43 | 0.45 | 0.61 | 0.94 |
| 100-180 Hz | 0.26 | 0.30 | 0.45 | 0.71 | 0.57 |
| 180-260 Hz (worker flight) | 0.069 | 0.061 | 0.65 | 0.76 | 0.91 |
| 260-400 Hz | 0.136 | 0.078 | 0.75 | 0.32 | 0.51 |
| 400-500 Hz | 0.044 | 0.034 | 0.65 | 0.56 | 0.91 |
| 500-600 Hz (swarming) | 0.027 | 0.013 | 0.75 | 0.18 | 0.34 |
| 600-1000 Hz | 0.061 | 0.036 | 0.70 | 0.44 | 0.63 |
| 1-4 kHz | 0.039 | 0.029 | 0.55 | 0.68 | 1.00 |

Values are shares of session power. Session AUC is the fraction of the
twenty queenright/queenless session pairs in which the queenright session
has the larger share. The last column drops CF001, a queenless session that
contributes six clips yet carries the same weight as a 1 200-clip session in
every session-level mean on this page. The 20-100 Hz gap, the largest in the
table, is mostly that one session. The swarming band, the best of the eight,
sits at p = 0.18 with all sessions and 0.34 without it. That is the same
finding as above, now with a number on it.

---

## 4. Three ways to split the data

![protocols](figures/03_protocols.png)

A small convolutional network on log-mel spectrograms, with a filterbank
that starts at 20 Hz so the low band is not discarded by a default. It is
trained the same way three times:

- **random** - five stratified folds over clips. Clips from the same
  ten-minute file land on both sides. This is how most published results are
  produced.
- **session** - leave one recording session out.
- **hive** - leave one hive out. This is the only protocol that answers the
  question a beekeeper asks.

Folds under the last two often contain a single class, so out-of-fold scores
are pooled and scored once.

| protocol | pooled AUC | balanced accuracy |
|---|---|---|
| random over clips | **0.9999** | 0.995 |
| leave one session out | **0.054** | 0.14 |
| leave one hive out | **0.375** | 0.39 |

The random number is real and means nothing. It measures whether the network
can tell which recording a clip came from. The session number is the
instructive one. An AUC of 0.05 is not a model that knows nothing. It is a
model that confidently assigns each new session the label of the session it
most resembles, and in this dataset the nearest session usually has the
other label (Hive1's two days, Hive3's three). A model that had learned
*queens* would sit near 0.5 on a held-out session at worst. This one learned
*sessions*, and it says so.

### 4.1 The same result, session by session

The pooled AUC hides which sessions are wrong. From the out-of-fold
probabilities of the leave-one-session-out run (`results/cv/`,
`src/session_checks.py`, `results/session_checks.json`; added 2 October):

| held-out session | truth | clips | called queenright | median p(queenright) |
|---|---|---|---|---|
| CF001 | queenless | 6 | 83 % | 0.99 |
| CF003 | queenright | 600 | 13 % | 0.01 |
| CJ001 | queenless | 374 | 81 % | 0.96 |
| GH001 | queenright | 600 | 21 % | 0.07 |
| Hive1, 31 May | queenless | 600 | 50 % | 0.50 |
| Hive1, 12 June | queenright | 600 | 0 % | 3e-07 |
| Hive3, 12 July | queenless | 600 | 99.8 % | 1.00 |
| Hive3, 15 July | queenless | 600 | 95 % | 1.00 |
| Hive3, 20 July | queenright | 321 | 0.6 % | 1e-05 |

No session is called right by a majority of its clips. Seven of the nine
are called wrong on more than 80 % of their clips, GH001 on 79 %, and the
ninth, Hive1 on 31 May, is a coin: 301 of 600. Ranked by mean probability,
every queenless session sits above every queenright one, a session-level
AUC of 0 out of 20 pairs. Only 2 of the 126 relabellings of nine sessions
are that extreme in either direction, p = 0.016. At the unit this page
insists on, the reversal is not noise.

Two things in that table change what was said above.

**The explanation covers five sessions, not nine.** "The nearest session
has the other label" can only be about Hive1 and Hive3, where the same hive
is in the training set on another day. The four citizen-science hives have
one session each, nothing of theirs is in training, and they are called
wrong just as firmly (5 of 6 clips, 87 %, 81 %, 79 %). It does not fit
Hive3 either: the queenless day of 12 July has a queenless neighbour three
days later and a queenright one eight days later, and is called queenright
on 99.8 % of its clips. The stored probabilities do not say which training
session a held-out one was matched to. The sentence was a guess and should
have been written as one.

**Chance is not 0.5 under this protocol.** Holding a queenright session out
leaves a training set that is 41-45 % queenright; holding a queenless one
out leaves 49-57 %. A model that learned nothing and returned its training
prior would score every queenright clip below every queenless one: pooled
AUC 0.00, not 0.50. The network's outputs are nowhere near those priors
(medians from 3e-07 to 1.00), so the prior alone is not what it is doing.
But some pull below 0.5 is built into leave-one-out with pooled scoring, and
its size cannot be read from these files. A run with the labels shuffled
across sessions would measure it, and that has not been done. "Near 0.5 at
worst" was too strong.

Leave-one-hive-out reads the same way. The network gives a held-out hive
one label whatever the day. Hive1 is called queenless on 99.8 % of its
queenless clips and on 100 % of its queenright ones (within-hive AUC 0.49).
Hive3 is called queenright on 75 % and 82 % (0.38). CF003 and GH001 are
called wrong on 76 % and 79 %, CF001 on 6 of 6, and CJ001 is the one hive
called right, on 65 %. The pooled 0.375 comes from which label each hive is
handed, not from any ordering within a hive.

---

## 5. A physical model of the hum

![wing-beat model](figures/04_wingbeat_model.png)

A hive sounds the way it does because tens of thousands of bees beat their
wings. Each bee is a periodic source: a stroke at a fundamental f0 that moves
with the bee's size, load and temperature. A periodic source has harmonics.
The colony spectrum is written as that superposition and nothing else:

    S(f) = g * [ sum_k p(f0_k) sum_h a(h, f0_k) L(f; h*f0_k, h*gamma)  +  B(f) ]

Here p(f0) is the distribution of wing-beat frequencies in the colony, a
55-point histogram from 80 to 350 Hz, one per clip. The harmonic profile of a
stroke, a(h, f0), is a small network of harmonic number and fundamental,
shared by every clip. L is a Lorentzian line whose width grows with harmonic
number. B(f) is a power-law background per clip for everything that is not a
wing beat. The model is fitted to the spectra alone, by log-spectral error.
**The queen label never enters the fit.** Under the session and hive
protocols the shared physics (line width, harmonic profile) is refitted on
the training sessions only and frozen before the held-out clips are
decomposed.

**What the fit finds.** A single-bee spectrum with a line width of 2.4 Hz at
the fundamental, and a harmonic profile that decays over five harmonics. The
learned colony distributions put most of the mass at 100-160 Hz, with the
textbook 180-260 Hz band nearly empty in every session. Where the mass sits
differs by session far more than by class. The queenless sessions include
one at 104 Hz and, on six clips, one at 245 Hz. (Section 5.1 checks this
paragraph session by session; the first sentence about the mass holds for
five sessions of nine.)

**What the quantities are worth.** Five values per clip go into a logistic
regression under the same three protocols: mean and spread of f0, share of
power in the comb, background slope and level. They are compared with the
eight band energies and with the network:

| reading of the sound | random | session | hive |
|---|---|---|---|
| log-mel CNN | 0.9999 | 0.054 | 0.375 |
| 8 band energies + logistic regression | 0.915 | 0.522 | 0.566 |
| wing-beat model, full p(f0) (55 numbers) | 0.909 | 0.203 | 0.387 |
| **wing-beat model, 5 named quantities** | 0.842 | **0.595** | **0.618** |

The pattern is the same in every row. The more freedom a reading has, the
better it does on the random split and the worse it does on a new colony.
The network is best on clips it has effectively seen and worst on colonies
it has not. The full wing-beat histogram, 55 numbers per clip, is enough to
identify sessions again and falls to 0.20. Only the five quantities with
physical names stay above chance on a new session (0.60) and a new hive
(0.62). That is modest, it rests on nine sessions, and it is not enough to
act on.

**The caveat that matters.** The model knows what a harmonic series looks
like. It does not know what a bee is. Any periodic source in the recording
is decomposed into "bees" at whatever fundamental fits: mains hum at 50 or
60 Hz and its harmonics, a fan, a compressor in the citizen-science
recordings. The mass at 100 and 150 Hz in several sessions (two, on a
closer look: section 5.1) is exactly where
the harmonics of a 50 Hz mains would sit. This is the same lesson as
section 4 in a different coat. The decomposition is only as good as the
recording. A physical model at least makes that failure visible - a
distribution parked at mains harmonics can be seen in the left panel - where
a spectrogram network hides it.

### 5.1 The distributions, session by session

Two sentences above describe nine distributions at once. Opened up, as
shares of p(f0) on the 55-point grid (`results/session_checks.json`; added
2 October):

| session | source | 80-95 Hz | 100-160 Hz | 180-260 Hz | p at 100 Hz | comb share |
|---|---|---|---|---|---|---|
| CF001 | OSBH | 0.21 | 0.10 | 0.03 | 0.00 | 0.003 |
| CF003 | OSBH | 0.30 | 0.18 | 0.21 | 0.05 | 0.74 |
| CJ001 | OSBH | 0.84 | 0.07 | 0.03 | 0.01 | 0.51 |
| GH001 | OSBH | 0.46 | 0.42 | 0.02 | 0.03 | 0.92 |
| Hive1, 31 May | NU-Hive | 0.16 | 0.69 | 0.13 | 0.04 | 0.67 |
| Hive1, 12 June | NU-Hive | 0.03 | 0.89 | 0.03 | 0.03 | 0.90 |
| Hive3, 12 July | NU-Hive | 0.01 | 0.83 | 0.02 | 0.00 | 0.97 |
| Hive3, 15 July | NU-Hive | 0.06 | 0.60 | 0.30 | 0.15 | 0.61 |
| Hive3, 20 July | NU-Hive | 0.12 | 0.55 | 0.26 | 0.21 | 0.47 |

**Correction: "most of the mass at 100-160 Hz" is true of the five NU-Hive
sessions (55-89 %) and of none of the four citizen-science hives (7-42 %).**
In those four the distribution is piled against the bottom of the grid. The
four lowest grid points, 80-95 Hz, hold 21-84 % of the mass, and the single
lowest point, 80 Hz, holds 30 % in CJ001 and 31 % in GH001. A distribution
leaning on the edge of its grid is asking for a fundamental the grid does
not offer. CJ001's mean of 104 Hz and GH001's 127 Hz mark where the grid
starts, not a wing-beat frequency. The 180-260 Hz band is nearly empty
(2-3 %) in five sessions, not in every one: it holds 13 %, 21 %, 26 % and
30 % in the other four.

**The mains suspicion narrows to two sessions.** A line at 100 Hz that
stands clear of its neighbours appears in Hive3 on 15 and 20 July: 15 % and
21 % of the mass in one grid point, 6.1 and 5.2 times the mean of the points
either side. In the other seven sessions the 100 Hz point holds 0-5 % and
0.2-1.6 times its neighbours. Hive3 on 12 July, the same hive three days
earlier, has 0.4 % there. So "several sessions" was two, and whatever puts a
line at 100 Hz was not there, or not audible, on 12 July. Those are also
the two sessions with the most mass in 180-260 Hz (30 % and 26 %).

**CF001 has no comb to describe.** On its six clips the model assigns 0.3 %
of the power to wing beats and the rest to the background. The 245 Hz
quoted above is the mean of a distribution fitted to almost nothing.

**The named quantities at the session level.** The 0.60 and 0.62 in the
table are pooled over clips. Four of the five quantities are stored as
session means (the background level is not), and they go through the same
exact test as the band table in section 3:

| quantity | queenright | queenless | session AUC | exact p | p without CF001 |
|---|---|---|---|---|---|
| mean f0 (Hz) | 146 | 159 | 0.45 | 0.74 | 0.66 |
| spread of f0 (Hz) | 57 | 54 | 0.65 | 0.83 | 0.54 |
| comb share | 0.76 | 0.55 | 0.65 | 0.36 | 0.63 |
| background slope | 1.40 | 2.03 | 0.35 | 0.36 | 0.69 |

None separates nine sessions (p from 0.36 up), and the two that come
closest lose even that without the six-clip session. Whatever the
regression finds across clips, it is not a difference between session
means.

**Which clips.** The network rows use 4 301 clips (at most 600 per
session). The three logistic-regression rows use 2 406 (at most 300 per
session, `MAX_CLIPS_PER_SESSION` in `src/wingbeat_pinn.py`). Each row is
comparable across its three protocols. The rows are not on the same clips.

---

## 6. Checks that pass

Eight checks, all passing:

- the queen label, hive and session are parsed from every file-name
  pattern in the dataset, and documentation files are rejected
- only segments annotated as hive sound are kept; a comma decimal is read
  correctly
- the mel filterbank covers every frequency from 40 Hz up, so the band
  where hive sound lives is not silently discarded
- a single bee in the wing-beat model produces peaks at f0, 2f0, 3f0, 4f0
  and nowhere else
- the line width at the third harmonic is three times the width at the
  first
- a colony placed entirely at 250 Hz is read back as f0 = 250 Hz with zero
  spread
- **the result is pinned**: the random-split AUC must exceed 0.99, the
  network's session and hive AUCs must be below 0.5, and the five wing-beat
  quantities must land between 0.5 and 0.8 on both leave-out protocols. The
  page is not allowed to claim a working detector.
- **no band separates the classes**: the exact permutation test over the
  126 relabellings of nine sessions returns p > 0.1 for every band, with
  and without the six-clip session. The estimator itself returns 1/126 on a
  perfectly separated case.

A second file, `tests/test_readme_numbers.py` (twelve checks, added
2 October), recomputes sections 4.1 and 5.1 from the stored outputs and
pins every number in them to `results/session_checks.json`. It needs no
audio.

---

## 7. Limits of the evidence

- **Nine sessions from six hives.** Every cross-colony number above has a
  sample size of six, not four thousand. The 0.60 / 0.62 could be 0.5 on the
  next six hives.
- **Two recording projects, two microphones, two countries.** Part of what
  separates sessions is equipment. A dataset with one microphone in twenty
  hives would answer the hive question far better than this one can.
- **Queen removal is not the only thing that changed** between the NU-Hive
  sessions. Date, weather, time of day and forage did too. The label is the
  queen; the sound is everything.
- **The wing-beat model is a spectral model, not a bee counter.** It cannot
  tell a bee from any other periodic source, and it does not model the
  box, which shapes the spectrum below 300 Hz.
- **No swarming, no disease, no robbing.** The dataset labels one state.

---

## 8. Notes from the build

This started as a dormant folder called `beehive-acoustics`. The code was
written on 7 September, the download stopped half way, and nothing was ever
analysed. Fetching the rest of the raw audio (3.7 GB) from Zenodo only
worked with resumable `curl -C -` and retries. The finished version, with
the wing-beat model added, went public on 12 September. The exact
permutation test in section 3 and the eighth check came later the same day.
Until then the band claim rested on eyeballing the spread. On 2 October
the pooled numbers of sections 4 and 5 were opened session by session
(4.1, 5.1). That cost the page one explanation and half of one sentence,
both marked where they stand. `CHANGELOG.md` keeps the dated list.

Two things I would not call finished. The global fit of the wing-beat model
reaches a log-spectral RMSE of 0.60 after 1 500 Adam steps, which is
mediocre. And the mains-harmonic problem in section 5 is stated, not
handled. Notching or masking the 50 Hz harmonics before the fit, and running
more steps, are the obvious next moves. For scale, `train.py` takes about
25 minutes on a laptop GPU and `wingbeat_pinn.py` about 20 minutes on a
laptop CPU.

---

## 9. Code and how to run it

The physics core is public in this repository:

- `src/prepare.py` - the recordings to labelled clips, with provenance
- `src/spectra.py` - session-level spectra and the band table
- `src/model.py` - the log-mel front end (filterbank written out) and the
  network
- `src/train.py` - the three protocols, pooled scoring, resumable
- `src/band_permutation.py` - the exact permutation test on the session
  band table (standard library only)
- `src/wingbeat_pinn.py` - the wing-beat mixture model, its named features,
  and the protocol comparison
- `src/session_checks.py` - sections 4.1 and 5.1: the stored outputs read
  one session at a time (numpy only, no audio)
- `tests/test_all.py` - the eight checks above
- `tests/test_readme_numbers.py` - sections 4.1 and 5.1 pinned to the
  results files

`results/` holds every number on this page as JSON. `data/SOURCE.md` says
how to fetch the recordings; they are not redistributed here.

```
pip install -r requirements.txt
python src/download.py && python src/prepare.py
python src/spectra.py && python src/band_permutation.py
python src/train.py                               # ~25 min on a laptop GPU
python src/wingbeat_pinn.py                       # ~20 min on a laptop CPU
python src/session_checks.py                      # seconds, no audio
python src/figures.py && python -m pytest tests
```

---

## 10. Licence and dataset credit

Documentation, figures and result files: CC BY 4.0. Source code in `src/`
and `tests/`: MIT.

The recordings belong to their authors and are distributed through Zenodo
under CC BY 4.0. Cite them, not this page:

- Inês Nolasco, Emmanouil Benetos. 2018. *To bee or not to bee: An annotated
  dataset for beehive sound recognition.* Zenodo, doi:10.5281/zenodo.1321278.
  Recordings from the NU-Hive project and the Open Source Beehives project.

---

*Other acoustic projects built on the same physics-first approach are listed
on the profile [README](https://github.com/drdmitrymikhaylov).*
