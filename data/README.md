# Data

dementia-typer does not include data. Git ignores every file in this folder except this README.

> **OASIS-3 is under a Data Use Agreement.** Register, accept the agreement and download the data
> yourself. Do not commit OASIS files, derived tables, model files trained on them or outputs that show rows.

## Source

| Item | Value |
|---|---|
| Name | OASIS-3: Open Access Series of Imaging Studies, longitudinal neuroimaging, clinical and cognitive dataset |
| Website | <https://www.oasis-brains.org> (access through the OASIS data portal after registration) |
| Terms | OASIS Data Use Agreement: research use, no redistribution, cite the OASIS-3 publication (LaMontagne et al., 2019) |
| Content used | Clinical visit (UDS) diagnoses and scores, FreeSurfer volumes of MR sessions, participant demographics |

## Expected files

Put three CSV files in one folder, for example `data/oasis3/`. Rename the export columns to the names below.

| File | Needed? | Columns |
|---|---|---|
| `clinical.csv` | Yes | `OASISID`, `days_to_visit` (or `OASIS_session_label` ending in `_dNNNN`), `age at visit`, `dx1`. Optional: `MMSE`, `memory`, `orient`, `judgment`, `commun`, `homehobb`, `perscare`, `CDRSUM`, `CDRTOT` |
| `freesurfer.csv` | No (feature sets need it) | `OASISID`, `days_to_scan` (or `MR_session` ending in `_dNNNN`), `IntraCranialVol`, `TotalGrayVol`, `CortexVol`, `CorticalWhiteMatterVol`, `SubCortGrayVol`, `Left-Hippocampus`, `Right-Hippocampus`, `Left-Lateral-Ventricle`, `Right-Lateral-Ventricle`, `CSF`. Optional: `FS QC Status` |
| `demographics.csv` | No | `OASISID`, `sex` (`M`/`F`), `education_years` |

## Procedure

1. Request access on the OASIS website and accept the Data Use Agreement.
2. Export the clinical diagnosis table, the FreeSurfer table and the demographics table.
3. Rename the columns to the names above and save the three CSV files in `data/oasis3/`.
4. Run `dementia-typer validate --data data/oasis3`. Extend the rules in `labels.py` for each `dx1` text that it lists.

## Synthetic data (no download)

`dementia-typer synth --out data/synthetic` writes the three files with the same columns. The values
are random. They are not OASIS data, and results on them say nothing about real participants.
