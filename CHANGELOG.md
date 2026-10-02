# Changelog

Dated notes on what changed on this page and why. Newest first.

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
