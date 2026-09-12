# Data source

Inês Nolasco, Emmanouil Benetos. 2018. *To bee or not to bee: An annotated
dataset for beehive sound recognition.* Zenodo, doi:10.5281/zenodo.1321278
(CC BY 4.0). Recordings from two projects:

- **NU-Hive** — two hives (Hive1, Hive3), each recorded on separate days with
  the queen present and after the queen was removed; ten-minute files, the
  queen status and the date in the file name.
- **Open Source Beehives (OSBH)** — citizen-science recordings from several
  hives, each labelled *Active* or *Missing Queen*; no session structure, the
  hive is the unit.

Every file has a `.lab` annotation marking which segments are hive sound
(`bee`) and which are external noise (`nobee`). Only `bee` segments are used.

The audio is not redistributed here. `src/download.py` fetches the record;
`src/prepare.py` cuts it into two-second clips at 8 kHz (`data/clips/`) and
writes `data/index.json` with the hive, session and queen label of every file.
Cite the authors above, not this page.
