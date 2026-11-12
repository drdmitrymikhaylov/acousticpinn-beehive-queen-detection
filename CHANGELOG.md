# Changelog

Dated notes on what changed on this page and why. Newest first.

## 2026-11-12

- New section 4.2 "What the pooled number is made of"
  (`src/pooled_decomposition.py`, `results/pooled_decomposition.json`).
  The leave-one-session-out AUC of 0.054 opened into its twenty
  session-pair cells (clip-weighted mean = the pooled number, asserted):
  none above 0.5, 16 below 0.1, median 0.008, largest 0.29 — the reversal
  holds pair by pair at the clip level, not only in the ranking of nine
  session means. CF001 carries 0.3 % of the pooled weight (one ninth of
  every session-level statistic); without it the number is 0.0536 to four
  decimals.
- The training-prior part of the pull below 0.5, which section 4.1 said
  could not be read from the files, can: removing each fold's prior shift
  (0.02 to 0.36 logits against network logits of 2.5 to 14.9) moves the
  pooled AUC from 0.054 to 0.064; reaching 0.5 would take 6.8 logits. The
  prior accounts for about 0.01 of AUC; the label-shuffled run is still
  the only way to size the rest.
- Leave-one-hive-out: 16 % of the pooled weight is the ordering within
  Hive1 and Hive3 (0.43), 84 % is which label a hive was handed (0.36);
  with the cross-hive pairs at chance the 0.375 would read 0.49. The
  citizen-science queenright hives and Hive3 score above Hive1's queenless
  day on 99–100 % of pairs and below Hive3's queenless days on 85–86 %:
  the network ranks hives. Finding 2 and the section 4.1 sentence amended.
- `results/cv/` now also holds the five random-split folds (20 files);
  the stored 0.9999 reproduces fold by fold.
- `tests/test_pooled_decomposition.py` (six checks) pins the section.

## 2026-10-19

- LICENSE now holds the standard MIT text and LICENSE-CONTENT the standard CC BY 4.0 text, so GitHub recognises the licence (it showed "Other" before). The terms are unchanged; the note on what each licence covers moved to `NOTICE.md`.

## 2026-10-02

- **Sections 4 and 5 opened session by session** (new 4.1 and 5.1,
  `src/session_checks.py`, `results/session_checks.json`, out-of-fold
  probabilities of the network in `results/cv/`). Under leave-one-session-out
  no session is called right by a majority of its clips: seven of nine are
  wrong on more than 80 %, one on 79 %, one is a coin (301 of 600). Ranked by
  mean probability every queenless session sits above every queenright one,
  which 2 of the 126 relabellings match (p = 0.016).
- **Withdrawn:** "the nearest session has the other label" as the
  explanation. It can only apply to Hive1 and Hive3; the four single-session
  hives are called wrong as firmly, and Hive3's queenless 12 July is called
  queenright on 99.8 % of clips with a queenless neighbour three days away.
- **Qualified:** "near 0.5 at worst". Leave-one-out changes the training
  prior (41-45 % queenright with a queenright session held out, 49-57 % with
  a queenless one), so a model that returned only its prior would have a
  pooled AUC of 0.00. How much of the reversal that explains is not known;
  a label-shuffled run would say.
- **Corrected:** "most of the mass at 100-160 Hz, 180-260 Hz nearly empty in
  every session". True of the five NU-Hive sessions (55-89 % in 100-160 Hz).
  In the four citizen-science hives 21-84 % of p(f0) sits on the four lowest
  grid points, so their mean f0 marks the grid floor, and 180-260 Hz holds
  13-30 % in four sessions.
- **Narrowed:** the 100 Hz line suspected of being mains is in two sessions,
  Hive3 on 15 and 20 July (15 % and 21 % of the mass, 6.1 and 5.2 times the
  neighbouring grid points), not "several".
- Session means of four named wing-beat quantities through the exact test:
  none separates the classes (p 0.36-0.83).
- Stated under the table: the network rows use 4 301 clips, the three
  logistic-regression rows 2 406.
- `tests/test_readme_numbers.py`: twelve checks pin 4.1 and 5.1 to the
  results files. Removed five macOS `._*` files that had been committed.

## 2026-09-12

- Exact permutation test on the nine-session band table (126 relabellings):
  the best band, 500-600 Hz, sits at p = 0.18, and 0.34 without the six-clip
  session. Eighth check added.
- First public version: three protocols, wing-beat mixture model.
