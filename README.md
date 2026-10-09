<div align="center">

# dementia-typer — Leakage-Aware Dementia Group Classification on OASIS-3

**dementia-typer is a research pipeline for analysts who study dementia groups in OASIS-3. It takes clinical visits and FreeSurfer volumes through these steps to a held-out, participant-level evaluation:**

`validate` → `join visits to MRI` → `map dx text` → `select by grouped CV` → `evaluate once` → `predict`.

![Groups](https://img.shields.io/badge/Groups-5-1F3864?style=for-the-badge)
![Feature sets](https://img.shields.io/badge/Feature_sets-A_B_C-2E5FD9?style=for-the-badge)
![CLI commands](https://img.shields.io/badge/CLI_commands-7-6E86E8?style=for-the-badge)
![Tests](https://img.shields.io/badge/Tests-55_passing-3DA35B?style=for-the-badge)
![Offline demo](https://img.shields.io/badge/Offline_demo-Yes-F5C542?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-A0399B?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-pipelines-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-tables-150458?style=flat-square&logo=pandas&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-v2_input_schema-E92063?style=flat-square&logo=pydantic&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-optional-555555?style=flat-square)
![Docs](https://img.shields.io/badge/Docs-ASD--STE100-5D6D7E?style=flat-square)

**[Summary](#1-summary)** ·
**[Workflow](#4-the-end-to-end-workflow)** ·
**[Run it](#14-how-to-run-dementia-typer)** ·
**[Configuration](#144-environment-variables)** ·
**[Known problems](#17-known-problems)** ·
**[Glossary](#19-glossary)**

</div>

> [!NOTE]
> This README uses ASD-STE100 Simplified Technical English. The writing rules and the project
> vocabulary are in [`docs/ste-style-guide.md`](docs/ste-style-guide.md). Each term in the
> [Glossary](#19-glossary) has only one meaning.

---

> [!WARNING]
> Do not use dementia-typer to diagnose a person. It is a research tool, not a medical device.
> A clinician must make every diagnosis. OASIS-3 comes from one research cohort, so the results can be biased for other populations.

dementia-typer classifies OASIS-3 visits into five groups: Normal, MCI, Alzheimer's, Vascular and Other dementia.
The main feature set uses only demographics and MRI volumes. The CDR scores that clinicians use to assign the diagnosis appear only in a reference "leakage ceiling".
Every split keeps the visits of one participant together, and the hold-out test participants are used once, after the model selection.
Each prediction goes through a pydantic schema and returns group names from one fixed class list.

This README is the **one location that explains all of dementia-typer**. It gives these topics:

- the general design
- each component and its procedure, step by step
- the decision rules
- the data map
- the runbook
- the validation results and the known problems

| If you are… | Read |
|---|---|
| A manager or reviewer | [1](#1-summary), [3](#3-design-rules), [4](#4-the-end-to-end-workflow), [16](#16-validation-results), [18](#18-key-points) |
| A developer who joins the project | All sections, in sequence. Keep [14](#14-how-to-run-dementia-typer) and [17](#17-known-problems) open while you work |
| An analyst who runs dementia-typer | [14](#14-how-to-run-dementia-typer), then the section for the component that you use |

---

## Table of contents

1. 🧭 [Summary](#1-summary)
2. 🏗️ [How dementia-typer is built](#2-how-dementia-typer-is-built)
   - 2.1 [Components](#21-components)
   - 2.2 [System context](#22-system-context)
   - 2.3 [Repository layout](#23-repository-layout)
3. 🛡️ [Design rules](#3-design-rules)
4. 🔄 [The end-to-end workflow](#4-the-end-to-end-workflow)
   - 4.1 [Full flow](#41-full-flow)
   - 4.2 [The life cycle of one visit](#42-the-life-cycle-of-one-visit)
   - 4.3 [Who does which step](#43-who-does-which-step)
5. 🔵 [Tables and the visit-to-MRI join](#5-tables-and-the-visit-to-mri-join)
6. 🏷️ [The diagnosis rules](#6-the-diagnosis-rules)
7. 🟢 [Feature sets](#7-feature-sets)
8. 🟣 [Participant splits and model selection](#8-participant-splits-and-model-selection)
9. 📏 [Evaluation and explanations](#9-evaluation-and-explanations)
10. 🔎 [The leak check](#10-the-leak-check)
11. 🧾 [Bundles, model cards and prediction](#11-bundles-model-cards-and-prediction)
12. ⚖️ [The decision rules](#12-the-decision-rules)
13. 🗂️ [Data and file map](#13-data-and-file-map)
14. ▶️ [How to run dementia-typer](#14-how-to-run-dementia-typer)
    - 14.1 [Prerequisites](#141-prerequisites) · 14.2 [Installation](#142-installation) · 14.3 [Run dementia-typer](#143-run-dementia-typer) · 14.4 [Environment variables](#144-environment-variables)
15. 🧩 [How to extend dementia-typer](#15-how-to-extend-dementia-typer)
16. ✅ [Validation results](#16-validation-results)
17. ⚠️ [Known problems](#17-known-problems)
18. 📌 [Key points](#18-key-points)
19. 📖 [Glossary](#19-glossary)
20. 📄 [License](#20-license)

---

## 1. Summary

**The problem.** OASIS-3 has many visits for each participant, free-text diagnoses and scores that clinicians use to make the diagnosis. These questions are difficult:

- How do you map the `dx1` texts to groups without wrong substring matches?
- How do you keep one participant out of both training and test?
- How much of the accuracy comes from CDR, which defines the diagnosis?
- How do you select a model without a look at the test set?
- How do you predict one visit without a hand-written feature order?

dementia-typer gives each of these questions its own component. Each component has a validated input and a tested output.

| Item | Value |
|---|---|
| Input | `clinical.csv` (required), `freesurfer.csv` and `demographics.csv` (optional) |
| Output | Model bundle, `metrics.json`, model card, ablation table, prediction JSON |
| Components | **11** modules: config, labels, data, features, modeling, evaluate, train, predict, explain, synthetic, plus the CLI |
| Groups | Normal, MCI, Alzheimer's, Vascular, Other dementia |
| Models | `majority` baseline, `logreg`, `forest` (default candidates), `boosting` (optional) |
| Offline mode | Synthetic tables, all commands and all tests. No download and no key |
| Safety | Participant-grouped splits. Selection on training participants only. CDR only in the ceiling set |
| Tests | **55** pass in CI (`.[dev]` only). 1 more test skips without the optional `shap` package |

```mermaid
flowchart LR
    IN["Clinical visits + FreeSurfer + demographics"] --> A["Validate and join"] --> B["Map dx text to group"]
    B --> C["Feature set A, B or C"] --> D["Grouped model selection"] --> E["One hold-out evaluation"] --> OUT["Bundle, metrics, model card"]
```

---

## 2. How dementia-typer is built

### 2.1 Components

| Component | Module | Purpose |
|---|---|---|
| Settings | `src/dementia_typer/config.py` | Environment variables: folders, seed, join window, number of folds |
| Diagnosis rules | `src/dementia_typer/labels.py` | `CLASSES`, ordered rules, unmapped report, encode and decode |
| Tables and join | `src/dementia_typer/data.py` | Schema checks, QC filter, visit-to-MRI join |
| Feature sets | `src/dementia_typer/features.py` | Features normalized by intracranial volume, sets `A`, `B`, `C` |
| Splits and selection | `src/dementia_typer/modeling.py` | Participant hold-out, pipelines, grids, inner search, nested CV |
| Metrics | `src/dementia_typer/evaluate.py` | Macro-F1, balanced accuracy, recall, AUROC, Brier, ECE, bootstrap |
| Experiments | `src/dementia_typer/train.py` | Experiment, leak check, bundle, model card |
| Prediction | `src/dementia_typer/predict.py` | Pydantic `Visit` schema and `predict_visit` |
| Explanations | `src/dementia_typer/explain.py` | SHAP values (optional extra) |
| Synthetic data | `src/dementia_typer/synthetic.py` | OASIS-3-like tables with progression and CDR leakage |
| CLI | `src/dementia_typer/cli.py` | `synth`, `validate`, `train`, `ablation`, `leakage`, `predict`, `demo` |

The component map shows which module calls which module. An arrow points from the caller to the module that it uses.

```mermaid
flowchart TB
    CLI["cli.py<br/>7 commands"]
    CFG["config.py<br/>Settings.from_env"]
    subgraph DATA["Data"]
        SYN["synthetic.py<br/>write"]
        DAT["data.py<br/>load_tables, join_visits, build_table"]
        LAB["labels.py<br/>CLASSES, RULES, map_diagnosis"]
        FEA["features.py<br/>make_features, FEATURE_SETS"]
    end
    subgraph LEARN["Training and evaluation"]
        TRN["train.py<br/>run_experiment, leakage_check, save, load"]
        MOD["modeling.py<br/>holdout_split, inner_search, nested_cv"]
        EVA["evaluate.py<br/>metrics, subject_bootstrap"]
    end
    subgraph USE["Use"]
        PRE["predict.py<br/>Visit, predict_visit"]
        EXP["explain.py<br/>shap_values, optional"]
    end

    CLI --> CFG
    CLI --> SYN
    CLI --> DAT
    CLI --> LAB
    CLI --> TRN
    CLI --> PRE
    DAT --> LAB
    FEA --> DAT
    TRN --> FEA
    TRN --> LAB
    TRN --> MOD
    TRN --> EVA
    EVA --> LAB
    PRE --> FEA
    PRE --> EVA
    PRE --> LAB
```

### 2.2 System context

```mermaid
flowchart TB
    U["Analyst"] --> CLI["dementia-typer CLI"]
    OASIS["OASIS-3 portal (manual download, data use agreement)"] --> D["data/ folder (git ignores it)"]
    SYN["Synthetic generator"] --> D
    D --> CLI
    CLI --> R["runs/ folder: bundle, metrics, model card"]
    J["Visit JSON"] --> CLI
```

### 2.3 Repository layout

```
dementia-typer/
├── .github/workflows/ci.yml       # CI: install .[dev], run pytest
├── data/README.md                 # source, terms, expected files, procedure
├── docs/ste-style-guide.md        # writing rules and project vocabulary
├── src/dementia_typer/
│   ├── config.py                  # settings from environment variables
│   ├── labels.py                  # classes and diagnosis rules
│   ├── data.py                    # schema checks and the visit-to-MRI join
│   ├── features.py                # feature sets A, B, C
│   ├── modeling.py                # splits, pipelines, inner search, nested CV
│   ├── evaluate.py                # metrics and participant bootstrap
│   ├── train.py                   # experiments, leak check, bundle, model card
│   ├── predict.py                 # pydantic schema and prediction
│   ├── explain.py                 # optional SHAP
│   ├── synthetic.py               # synthetic tables
│   └── cli.py                     # command-line interface
├── tests/                         # 56 tests (1 needs the optional shap package)
├── .env.example                   # variable names only
└── pyproject.toml                 # package, extras, console script
```

---

## 3. Design rules

### 3.1 CDR is a ceiling, not a feature of the main model

Clinicians use the CDR boxes, `CDRSUM` and `CDRTOT` to assign the diagnosis. `features.py` puts them only in feature set `C`. The model card of a set `C` bundle has a warning that its result is a leakage ceiling.

```mermaid
flowchart LR
    A["Set A<br/>demographics + 10 MRI columns"] --> B["Set B<br/>A + MMSE"]
    B --> C["Set C<br/>B + 6 CDR boxes, CDRSUM, CDRTOT"]
    A --> MAIN[/"Main model result"/]
    B --> REF[/"Result with a short cognitive test"/]
    C --> CHK{"LEAKAGE_FEATURES<br/>in the features?"}
    CHK -- "yes" --> WARN[/"Model card warning:<br/>leakage ceiling, not a valid estimate"/]
```

### 3.2 One participant, one side of each split

`modeling.holdout_split` holds out whole participants, stratified by the group of their last visit. The inner search uses `GroupKFold` on `OASISID`. `assert_disjoint` stops the run if a participant is on both sides.

### 3.3 The test set never selects a model

`inner_search` selects the candidate and its parameters by the inner grouped macro-F1 on the training participants. The hold-out test is used once, after the selection. `--nested` adds an outer grouped CV that estimates the error of the full selection procedure.

### 3.4 The preprocessor is inside the pipeline

The imputer, the scaler and the selector are steps of one scikit-learn `Pipeline`. Each fit uses only the rows of its fold. Permutation importance uses the hold-out test participants.

### 3.5 One class list

`labels.CLASSES` fixes the order of the five groups. The bundle stores the class list, and `load` refuses a bundle with a different list. Predictions return group names, never bare integers.

### 3.6 Explicit diagnosis rules

`labels.RULES` is an ordered list of word-boundary patterns. "Non AD dementia" maps to Other dementia, not to Alzheimer's. A text that no rule matches gives no group and appears in the `validate` output.

### 3.7 Validated input for prediction

`predict.Visit` is a pydantic model with named fields, ranges and `extra="forbid"`. The bundle gives the feature order. The output lists the features that the imputer filled.

---

## 4. The end-to-end workflow

### 4.1 Full flow

```mermaid
flowchart TD
    C[/"clinical.csv"/] --> VC["validate_clinical"]
    M[/"freesurfer.csv"/] --> VM["validate_mri + QC filter"]
    D[/"demographics.csv"/] --> VD["validate_demographics"]
    VC --> J["join_visits (nearest MR session in window)"]
    VM --> J
    VD --> J
    J --> L["map_diagnosis: dx1 to group"]
    L --> UM{"validate: any dx1 text<br/>with no rule?"}
    UM -- "yes, exit code 2" --> HR{{"HUMAN<br/>review the text, extend labels.RULES"}}
    L -- "visits with a group" --> F["make_features + feature set"]
    F --> H["holdout_split by participant"]
    H --> IS["inner_search (GroupKFold, macro-F1)"]
    H --> NC["nested_cv (optional)"]
    IS --> EV["evaluate once on hold-out participants"]
    EV --> B[("runs/set_X/<br/>model.joblib, metrics.json, model_card.md")]
    VJ[/"Visit JSON"/] --> P
    B --> P["predict_visit (pydantic Visit)"]
    P --> OUT[/"Group probabilities and imputed features"/]
    OUT --> HC{{"HUMAN<br/>a clinician makes the diagnosis"}}

    classDef human fill:#fff3cd,stroke:#b8901f,color:#3d2f00,font-weight:bold
    class HR,HC human
```

### 4.2 The life cycle of one visit

```mermaid
stateDiagram-v2
    state "Raw visit row" as Raw
    state "Validated visit" as Valid
    state "Visit with MRI volumes" as WithMRI
    state "Visit with empty volumes" as NoMRI
    state "Visit with a group" as Grouped
    state "Left out, no group" as Unmapped
    state "Feature row" as Featured
    state "Training participant" as Train
    state "Hold-out participant" as Test
    state "Five group probabilities" as Scored
    [*] --> Raw
    Raw --> SchemaError: column absent, visit not unique, MMSE out of range
    Raw --> Valid: validate_clinical
    Valid --> WithMRI: passed MR session within the window
    Valid --> NoMRI: no MR session within the window
    WithMRI --> Grouped: map_diagnosis
    NoMRI --> Grouped: map_diagnosis
    WithMRI --> Unmapped: no rule matches dx1
    NoMRI --> Unmapped: no rule matches dx1
    Grouped --> Featured: make_features
    Featured --> Train: holdout_split, with its participant
    Featured --> Test: holdout_split, with its participant
    Train --> Train: inner_search selects the candidate
    Test --> Scored: selected pipeline, once
    Scored --> [*]
    Train --> [*]
    Unmapped --> [*]
    SchemaError --> [*]
```

1. `validate_clinical` reads the visit row and checks the required columns, the key and the MMSE range.
2. `join_visits` attaches the nearest passed MR session of the same participant within 365 days.
3. `map_diagnosis` changes the `dx1` text into a group. If no rule matches, the visit leaves the analysis.
4. `make_features` divides each volume by the intracranial volume and adds the demographic values.
5. The split puts the visit, with all other visits of its participant, in training or in the hold-out test.
6. If the visit is in training, it helps the inner search to select a candidate.
7. If the visit is in the test, the selected pipeline gives five group probabilities for it.

### 4.3 Who does which step

```mermaid
sequenceDiagram
    autonumber
    actor AN as Analyst
    participant CLI as dementia-typer CLI
    participant DAT as data.py
    participant TRN as train.py
    participant MOD as modeling.py
    participant FS as runs/ folder
    participant PRE as predict.py

    AN->>CLI: dementia-typer train --data data/oasis3 --feature-set A --nested
    CLI->>CLI: Settings.from_env
    CLI->>DAT: build_table(data, join_window_days)
    DAT->>DAT: load_tables, validate_*, join_visits, map_series
    DAT-->>CLI: visit table with group
    CLI->>TRN: run_experiment(table, A, models, selectors, k, seed)
    TRN->>MOD: holdout_split by participant
    TRN->>MOD: nested_cv on training participants (--nested)
    TRN->>MOD: inner_search: GridSearchCV with GroupKFold
    MOD-->>TRN: best pipeline, chosen candidate, search log
    TRN->>TRN: metrics on hold-out, subject_bootstrap, majority baseline, permutation importance
    TRN-->>CLI: Experiment
    CLI->>FS: save: model.joblib, metrics.json, model_card.md
    CLI-->>AN: test metrics and paths
    AN->>CLI: dementia-typer predict --model model.joblib --input visit.json
    CLI->>TRN: load: check the class list
    CLI->>PRE: predict_visit(bundle, payload)
    PRE->>PRE: Visit.model_validate, make_features in bundle order
    PRE-->>CLI: prediction, probabilities, imputed_features, note
    CLI-->>AN: JSON result
    AN->>AN: a clinician makes the diagnosis
```

---

## 5. Tables and the visit-to-MRI join

**Purpose.** Make one row for each clinical visit with the MRI volumes of a close MR session.

```mermaid
flowchart TD
    CL[/"clinical.csv"/] --> DV{"days_to_visit column?"}
    DV -- "no" --> LBL["day_from_label:<br/>suffix _dNNNN"]
    DV -- "yes" --> VC
    LBL --> VC["validate_clinical: required columns,<br/>unique visits, MMSE 0 to 30"]
    FS[/"freesurfer.csv, optional"/] --> VM["validate_mri: required columns,<br/>IntraCranialVol positive"]
    VM --> QC{"FS QC Status column?"}
    QC -- "yes" --> QF["Keep passed, pass, ok"]
    QC -- "no" --> J
    QF --> J["merge_asof by OASISID: nearest session,<br/>tolerance DEMENTIA_JOIN_WINDOW_DAYS"]
    VC --> J
    J --> GAP["scan_gap_days"]
    DM[/"demographics.csv, optional"/] --> VD["validate_demographics:<br/>unique OASISID"]
    GAP --> MD["Merge demographics on OASISID"]
    VD --> MD
    MD --> MAP["map_series: dx1 to group"]
    MAP --> OUT[/"Visit table"/]
    VC -- "problem" --> ERR[/"SchemaError, exit code 1"/]
```

| Input | Output |
|---|---|
| Clinical, FreeSurfer and demographics tables | Visit table with volumes, `scan_gap_days`, demographics and `group` |

**Procedure**

1. If `days_to_visit` is missing, read the day from `OASIS_session_label` (suffix `_dNNNN`).
2. Check the required columns, unique visits and the MMSE range (0–30).
3. Keep only MR sessions with a passed `FS QC Status`, if the column exists.
4. For each visit, take the MR session of the same participant with the smallest day distance, if it is not more than `DEMENTIA_JOIN_WINDOW_DAYS`.
5. Add the demographics of the participant and map the `dx1` text to a group.

**Rules**

- A join never crosses participants.
- A visit with no MR session in the window keeps empty volumes. The imputer of the pipeline fills them.

---

## 6. The diagnosis rules

**Purpose.** Map each `dx1` text to one group, or to no group.

```mermaid
flowchart TD
    IN[/"dx1 text"/] --> E{"Empty or missing?"}
    E -- "yes" --> NONE[/"No group:<br/>listed by validate"/]
    E -- "no" --> NORM["normalize: strip, lower case,<br/>one space"]
    NORM --> R1{"Rule 1: MCI patterns?"}
    R1 -- "yes" --> G1[/"MCI"/]
    R1 -- "no" --> R2{"Rule 2: Normal patterns?"}
    R2 -- "yes" --> G2[/"Normal"/]
    R2 -- "no" --> R3{"Rule 3: non ad?"}
    R3 -- "yes" --> G3[/"Other dementia"/]
    R3 -- "no" --> R4{"Rule 4: vascular, vad?"}
    R4 -- "yes" --> G4[/"Vascular"/]
    R4 -- "no" --> R5{"Rule 5: ad, dat, alzheimer?"}
    R5 -- "yes" --> G5[/"Alzheimer's"/]
    R5 -- "no" --> R6{"Rule 6: other dementia patterns?"}
    R6 -- "yes" --> G3
    R6 -- "no" --> NONE
```

| Order | Group | Pattern (on lower-case text) | Examples |
|---|---|---|---|
| 1 | MCI | `uncertain`, `incipient`, `0.5 in memory`, `ques. impair`, `questionable`, `impair reversible`, `mci`, `mild cognitive`, `w/o dement` | "0.5 in memory only", "uncertain- possible NON AD dem" |
| 2 | Normal | `cognitively normal`, `no dementia`, `normal` (the full text only) | "Cognitively normal" |
| 3 | Other dementia | `non ad`, `non-ad`, `nonad` | "Non AD dem- Other primary" |
| 4 | Vascular | `vascular`, `vad` | "Vascular Demt- primary" |
| 5 | Alzheimer's | `ad`, `dat`, `alzheimer` (whole words) | "AD Dementia", "DAT" |
| 6 | Other dementia | `dlbd`, `lewy`, `frontotemporal`, `ftd`, `pdd`, `parkinson`, `dementia/pd`, `huntington`, `prion`, `alcohol`, `other primary`, a word that starts with `dem` | "DLBD- primary" |

The first rule that matches wins. The rules use word boundaries, so "headache" does not match `ad`. `validate` returns exit code 2 and lists each text with no rule.

---

## 7. Feature sets

```mermaid
flowchart LR
    T[/"Visit table"/] --> AGE["age = age at visit"]
    T --> SEX["sex_male: M 1, F 0, else NaN"]
    T --> EDU["education_years"]
    T --> ICV["icv_l = IntraCranialVol / 1,000,000"]
    T --> VOL["9 volumes: volume / IntraCranialVol<br/>x 1000, per mille"]
    T --> COG["MMSE and CDR columns as numbers,<br/>NaN if absent"]
    AGE --> MF["make_features"]
    SEX --> MF
    EDU --> MF
    ICV --> MF
    VOL --> MF
    COG --> MF
    MF --> CF{"columns_for: set A, B or C"}
    CF -- "other value" --> ERR[/"ValueError"/]
    CF -- "A, B or C" --> X[/"Feature matrix X"/]
```

| Set | Columns | Use |
|---|---|---|
| `A` | `age`, `sex_male`, `education_years`, `icv_l`, 9 volumes divided by intracranial volume (per mille) | Main model: an objective signal with no cognitive test |
| `B` | `A` + `MMSE` | A short cognitive test added |
| `C` | `B` + `memory`, `orient`, `judgment`, `commun`, `homehobb`, `perscare`, `CDRSUM`, `CDRTOT` | Leakage ceiling for reference only |

The 9 volumes: `TotalGrayVol`, `CortexVol`, `CorticalWhiteMatterVol`, `SubCortGrayVol`, `Left-Hippocampus`, `Right-Hippocampus`, `Left-Lateral-Ventricle`, `Right-Lateral-Ventricle`, `CSF`.

---

## 8. Participant splits and model selection

**Purpose.** Select one pipeline with the training participants only, and keep the test participants for one final measurement.

```mermaid
flowchart TD
    IN[/"Labelled visits, X, y, OASISID"/] --> SL["subject_labels: group of the last visit"]
    SL --> RS["Groups with fewer than 5 participants<br/>share one stratum"]
    RS --> HO["StratifiedKFold of participants,<br/>first fold is the hold-out, about 20%"]
    HO --> DJ{"assert_disjoint:<br/>participant on both sides?"}
    DJ -- "yes" --> LE[/"LeakageError"/]
    DJ -- "no" --> NS{"--nested?"}
    NS -- "yes" --> NCV["nested_cv: StratifiedGroupKFold outer folds,<br/>inner_search with 3 folds in each"]
    NS -- "no" --> IS
    NCV --> IS["inner_search: for each candidate,<br/>GridSearchCV with GroupKFold, f1_macro"]
    IS --> BEST["Highest inner macro-F1,<br/>refit on all training participants"]
    BEST --> TEST["Predict the hold-out<br/>participants once"]
    TEST --> OUT[/"Experiment"/]
```

| Input | Output |
|---|---|
| Feature table, groups, `OASISID` | Selected pipeline, search log, optional nested CV scores |

**Procedure**

1. Give each participant the group of the last visit. Groups with fewer than 5 participants share one stratum.
2. Put one fold of a 5-fold stratified split of the participants in the hold-out test (about 20%).
3. For each candidate (model × selector), run a grid search with `GroupKFold` and macro-F1.
4. Keep the candidate with the highest inner macro-F1, refit on all training participants.
5. With `--nested`, repeat steps 3 and 4 inside an outer 5-fold grouped CV of the training participants.

| Model | Estimator | Grid |
|---|---|---|
| `majority` | `DummyClassifier(most_frequent)` | none (baseline) |
| `logreg` | `LogisticRegression(class_weight="balanced")` | `C` in 0.1, 1, 10 |
| `forest` | `RandomForestClassifier`, 200 trees, `balanced_subsample` | `max_depth` in None, 8. `min_samples_leaf` in 1, 5 |
| `boosting` | `HistGradientBoostingClassifier`, `class_weight="balanced"` | `learning_rate` in 0.05, 0.1. `max_leaf_nodes` in 15, 31 |

| Selector | Step |
|---|---|
| `none` | no selection |
| `mi` | `SelectKBest` with mutual information, `k` features (default 8) |
| `rfe` | `RFE` with a logistic regression, `k` features |

---

## 9. Evaluation and explanations

```mermaid
flowchart TD
    P[/"Pipeline probabilities"/] --> FP["full_proba: columns in CLASSES order,<br/>0 for a class the model never saw"]
    Y[/"True groups of the test visits"/] --> M["metrics"]
    FP --> M
    M --> PRES["macro_f1 and balanced_accuracy<br/>on the groups that occur"]
    M --> RC["recall, precision, support,<br/>absent_classes, confusion_matrix"]
    M --> CAL["brier, top-label ece with 10 bins"]
    M --> AUC["auroc_ovr for each group with<br/>both values, macro_auroc_ovr"]
    FP --> BS["subject_bootstrap: draw whole participants,<br/>200 draws, 2.5 and 97.5 percentiles"]
    FP --> PI["permutation_importance on the test<br/>participants, f1_macro, 3 repeats"]
    PRES --> OUT[/"test, ci, importance"/]
    RC --> OUT
    CAL --> OUT
    AUC --> OUT
    BS --> OUT
    PI --> OUT
```

| Metric | Meaning |
|---|---|
| `macro_f1` | Mean F1 over the groups that occur in the test. The main metric |
| `balanced_accuracy` | Mean recall over the groups that occur |
| `recall`, `precision` | For each group. Recall is `NaN` for a group with no test visit |
| `absent_classes` | Groups with no test visit |
| `macro_auroc_ovr` | Mean one-vs-rest AUROC |
| `brier` | Multi-class Brier score |
| `ece` | Top-label expected calibration error, 10 bins |
| `confusion_matrix` | Rows: true group. Columns: predicted group. Order of `CLASSES` |
| CI | 95% participant bootstrap interval of macro-F1 and balanced accuracy (default 200 draws) |

Permutation importance shuffles one feature at a time on the test participants and measures the drop in macro-F1 (3 repeats). With the `explain` extra, `explain.shap_values` gives SHAP values for held-out visits.

---

## 10. The leak check

`leakage_check` trains one fixed forest on feature set `A` two times:

```mermaid
flowchart LR
    T[/"Labelled visits, set A"/] --> VS["train_test_split by visit,<br/>stratified, 20% test"]
    T --> PS["holdout_split by participant"]
    VS --> F1["Forest fit and macro-F1"]
    PS --> F2["Forest fit and macro-F1"]
    F1 --> GAP["gap = visit_split - participant_split"]
    F2 --> GAP
    GAP --> OUT[/"visit_split, participant_split, gap"/]
```

1. With a visit-level stratified split, where one participant can be in training and test.
2. With the participant-level hold-out split.

The difference of the two macro-F1 values shows how much a visit-level split overstates the result. A model can know a participant again from the fixed brain size of that participant.

---

## 11. Bundles, model cards and prediction

`train.save` writes three files:

```mermaid
flowchart TD
    EXP[/"Experiment"/] --> SV["train.save"]
    SV --> B[("model.joblib, metrics.json,<br/>model_card.md")]
    B --> LD["train.load"]
    LD --> CL{"Same CLASSES list?"}
    CL -- "no" --> E1[/"ValueError, exit code 1"/]
    CL -- "yes" --> V{"Visit.model_validate:<br/>known fields, values in range?"}
    J[/"Visit JSON"/] --> V
    V -- "no" --> E2[/"ValidationError, exit code 1"/]
    V -- "yes" --> MF["make_features, columns<br/>in the bundle order"]
    MF --> IM["imputed_features: empty columns"]
    MF --> PR["pipeline.predict_proba, full_proba"]
    PR --> OUT[/"prediction, 5 probabilities,<br/>feature_set, note"/]
    IM --> OUT
    OUT --> HUMAN{{"HUMAN<br/>a clinician makes the diagnosis"}}

    classDef human fill:#fff3cd,stroke:#b8901f,color:#3d2f00,font-weight:bold
    class HUMAN human
```

| File | Contents |
|---|---|
| `model.joblib` | Pipeline, feature list, `CLASSES`, feature set, chosen candidate, package version |
| `metrics.json` | Search log, test metrics, baseline metrics, CIs, nested scores, importance |
| `model_card.md` | Intended use, features, split sizes, metrics, warning for set `C` |

**Prediction procedure**

1. Read a JSON object. Field names are the column names (for example `age at visit`, `Left-Hippocampus`, `MMSE`).
2. Validate it with `Visit`. Unknown fields and values out of range give an error and exit code 1.
3. Make the features in the order of the bundle.
4. Return the predicted group, the five probabilities, the filled features and a note that a clinician must make the diagnosis.

---

## 12. The decision rules

| Value | Where | Number |
|---|---|---|
| Join window | `DEMENTIA_JOIN_WINDOW_DAYS` | 365 days |
| QC values kept | `data.QC_PASSED` | `passed`, `pass`, `ok` |
| MMSE range | `validate_clinical`, `Visit` | 0–30 |
| Age range for prediction | `Visit.age` | 18–110 years |
| CDR box range | `Visit` | 0–3 (`CDRSUM` 0–18) |
| Hold-out size | `holdout_split(test_fraction=...)` | 0.2 of the participants |
| Rare stratum | `holdout_split` | fewer than 5 participants in a group |
| Inner folds | `DEMENTIA_N_SPLITS` | 5 (minimum 2) |
| Selector size | `--k` | 8 features |
| Bootstrap draws | `--bootstrap` | 200 |

---

## 13. Data and file map

| Path | Committed? | Contents |
|---|---|---|
| `data/README.md` | Yes | Source, terms, expected files, procedure |
| `data/oasis3/*.csv` | No (git ignores `/data/*`) | OASIS-3 exports (data use agreement) |
| `data/synthetic/*.csv` | No | Output of `dementia-typer synth` |
| `runs/set_<A/B/C>/model.joblib` | No | Bundle |
| `runs/set_<A/B/C>/metrics.json` | No | Metrics and search log |
| `runs/set_<A/B/C>/model_card.md` | No | Model card |
| `.env` | No | Local settings |

Do not commit a bundle trained on OASIS-3. It is derived from data under the data use agreement.

---

## 14. How to run dementia-typer

### 14.1 Prerequisites

| Need | For |
|---|---|
| Python 3.11+ | All components |
| OASIS-3 access | Real results (see [`data/README.md`](data/README.md)). The demo and the tests do not need it |
| `shap` (extra `explain`) | SHAP values only |

### 14.2 Installation

```bash
git clone https://github.com/KrishnaAnnavaram/dementia-typer.git
cd dementia-typer
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"         # add ,explain for SHAP
```

### 14.3 Run dementia-typer

Offline demo (synthetic data, about three minutes on a laptop):

```mermaid
flowchart LR
    S["synthetic.write<br/>300 participants, seed 0,<br/>temporary folder"] --> V["cmd_validate"]
    V --> B["build_table"]
    B --> L["leakage_check, set A"]
    L --> E["run_experiment for sets A, B, C,<br/>200 bootstrap draws"]
    E --> P[/"Metrics of each set on the console"/]
```

```bash
dementia-typer demo
```

Step by step:

```bash
dementia-typer synth --out data/synthetic --subjects 300 --seed 0
dementia-typer validate --data data/synthetic
dementia-typer leakage --data data/synthetic
dementia-typer train --data data/synthetic --feature-set A --nested --out runs/set_A
dementia-typer ablation --data data/synthetic
dementia-typer predict --model runs/set_A/model.joblib --input visit.json
```

Example `visit.json`:

```json
{"age at visit": 68, "sex": "F", "education_years": 16, "IntraCranialVol": 1400000,
 "Left-Hippocampus": 3800, "Right-Hippocampus": 3900, "MMSE": 30}
```

| Command | What it does | Exit code |
|---|---|---|
| `synth` | Writes synthetic `clinical.csv`, `freesurfer.csv`, `demographics.csv` | 0 |
| `validate` | Checks the tables and the join, counts the groups, lists dx texts with no rule | 0 OK, 2 texts with no rule, 1 error |
| `train` | Hold-out, inner search, optional nested CV, test, bundle and model card | 0 |
| `ablation` | Runs feature sets `A`, `B`, `C` and prints the metrics of each | 0 |
| `leakage` | Visit-level vs. participant-level macro-F1 | 0 |
| `predict` | Validates one visit JSON and prints group probabilities | 0, 1 for a schema error |
| `demo` | Synthetic data, validation, leak check, sets `A`, `B`, `C` | 0 |

### 14.4 Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `DEMENTIA_DATA_DIR` | `config.Settings` | Data folder in `Settings.data_dir` (default `data/oasis3`). No command reads it: each command that reads data needs `--data` |
| `DEMENTIA_OUTPUT_DIR` | `train` | Run folder (default `runs`) |
| `DEMENTIA_SEED` | splits, models, bootstrap | Seed (default 42) |
| `DEMENTIA_JOIN_WINDOW_DAYS` | `join_visits` | Maximum visit-to-scan distance (default 365) |
| `DEMENTIA_N_SPLITS` | inner and nested CV | Number of grouped folds (default 5, minimum 2) |

dementia-typer needs no credentials. `.env.example` gives the variable names. Git ignores `.env`. dementia-typer does not load `.env` itself. Set the variables in the shell before you run a command.

---

## 15. How to extend dementia-typer

| You want to… | Do this | Code change? |
|---|---|---|
| Map a new dx text | Add a pattern to `labels.RULES` and a test case in `tests/test_labels.py` | Small |
| Add a volume | Add the column to `data.MRI_VOLUMES` and a field to `predict.Visit` | Small |
| Add a feature set | Add an entry to `features.FEATURE_SETS` and `SET_DESCRIPTIONS` | Small |
| Add a model | Add a branch to `build_pipeline` and a grid to `GRIDS` | Small |
| Use the boosting model | `train --models logreg forest boosting` | No |
| Add calibration | Wrap the selected pipeline in `CalibratedClassifierCV` with grouped folds | Yes |

---

## 16. Validation results

All numbers below come from this repository. The model numbers use **synthetic data** (`dementia-typer demo`, 300 participants, seed 0). They do not describe OASIS-3 participants.

| Validation | Result | Command |
|---|---|---|
| Unit tests | CI installs only `.[dev]`: **55 passed**, 1 skipped (`explain` extra, SHAP) | `pip install -e ".[dev]" && pytest -q` |
| Join and rules (synthetic) | 879 visits, 300 participants, 724 visits with an MR session in 365 days, 0 unmapped texts | `dementia-typer validate` |
| Leak check, forest macro-F1 (synthetic) | Visit split 0.711, participant split 0.628, gap 0.082 | `dementia-typer leakage` |
| Split sizes (synthetic) | 240 training and 60 test participants (705 and 174 visits) | `dementia-typer train` |
| Nested CV, set `A` (synthetic) | Mean macro-F1 0.611 over 5 outer folds of the training participants | `dementia-typer train --nested` |
| Top permutation importance, set `A` (synthetic) | `CorticalWhiteMatterVol_icv` 0.191, `Right-Hippocampus_icv` 0.136, `Left-Hippocampus_icv` 0.098 (drop in macro-F1) | `dementia-typer train` |

Hold-out test on synthetic data (participant split, majority baseline macro-F1 0.139, accuracy 0.534):

| Feature set | Selected | Macro-F1 (95% participant CI) | Balanced accuracy | Accuracy | Macro AUROC | ECE |
|---|---|---|---|---|---|---|
| `A` demographics + MRI | `logreg`, no selector | 0.640 (0.522–0.757) | 0.683 | 0.678 | 0.880 | 0.096 |
| `B` + MMSE | `forest`, no selector | 0.858 (0.691–0.957) | 0.841 | 0.920 | 0.981 | 0.109 |
| `C` + CDR (ceiling) | `logreg`, no selector | 0.911 (0.794–1.000) | 0.909 | 0.971 | 0.999 | 0.034 |

The generator makes CDR follow the group almost exactly. Thus set `C` is close to perfect for Normal, MCI and Alzheimer's, and this shows the leakage. CDR does not separate the dementia types, so the recall of Other dementia stays at 0.55 in all sets.
The leak check shows that a visit-level split gives a higher macro-F1 than a participant split.
These numbers prove that the pipeline and its checks work. They do not give the accuracy on OASIS-3.

The earlier prototype reported about 96% accuracy on OASIS-3. That number is a prototype result, not reproduced here. It used CDR features and a visit-level split, so this project does not compare with it.

---

## 17. Known problems

Read these problems before you use dementia-typer for a publication.

| # | Area | Problem | Impact and action |
|---|---|---|---|
| 1 | Real data | CI runs on synthetic data only. Results on OASIS-3 are not reproduced in CI | Run `ablation` on your OASIS-3 export and review each table |
| 2 | Diagnosis rules | The rules cover the common `dx1` texts. OASIS-3 has more texts, and some are ambiguous ("uncertain, possible AD") | Run `validate` and extend `labels.RULES`. Have a clinician review the map |
| 3 | Small groups | Vascular and Other dementia have few participants. Their CIs are wide | Report the CI and the support of each group |
| 4 | Calibration | The selected pipeline is not calibrated. ECE is 0.03–0.11 on synthetic data | Do not read the probabilities as risks |
| 5 | Cohort bias | OASIS-3 participants are volunteers from one research center | Do not generalize to other populations without external validation |
| 6 | Column names | The FreeSurfer export uses other column names in some releases | Rename the columns to the names in `data/README.md` |
| 7 | Boosting speed | `HistGradientBoostingClassifier` with 5 classes and a full grid is slow on a CPU | It is not a default candidate. Add it with `--models` when you have the time |
| 8 | Visit order | Visits of one participant are not independent. The bootstrap resamples participants, but the metrics weight participants by their visit count | Report a last-visit evaluation for each participant if you need one weight for each person |
| 9 | Inner CV | The grid search uses `GroupKFold` without stratification. A small group can be absent from an inner fold | Use fewer folds for small datasets |

---

## 18. Key points

1. **The main model does not see CDR.** CDR appears only in the leakage ceiling set `C`, with a warning in the model card.
2. **Participants never cross a split.** The hold-out, the inner search and the nested CV group by `OASISID`.
3. **The test set measures, it does not select.** The inner grouped macro-F1 selects the candidate.
4. **One class list from training to prediction.** The bundle stores `CLASSES`, and predictions return names.
5. **Inputs are validated.** A pydantic schema checks each prediction input, and a schema check covers each table.
6. **All of it runs offline.** Synthetic tables feed the demo and the 55 tests.

---

## 19. Glossary

| Term | Meaning |
|---|---|
| **OASIS-3** | Open Access Series of Imaging Studies, a longitudinal dataset of aging and dementia |
| **participant** | One person, identified by `OASISID` |
| **visit** | One clinical visit of a participant |
| **MR session** | One MRI scan with FreeSurfer volumes |
| **join window** | The maximum distance in days between a visit and its MR session |
| **group** | One of the 5 classes: Normal, MCI, Alzheimer's, Vascular, Other dementia |
| **MCI** | Mild cognitive impairment: impairment without a confirmed dementia |
| **CDR** | Clinical Dementia Rating: six boxes, the sum (`CDRSUM`) and the global score (`CDRTOT`) |
| **MMSE** | Mini-Mental State Examination, 0 to 30 |
| **feature set** | `A`, `B` or `C`: the feature columns of an experiment |
| **leakage ceiling** | The result of set `C`, which contains the scores that define the diagnosis |
| **candidate** | One model and selector pair |
| **inner search** | The grouped grid search on the training participants |
| **nested CV** | An outer grouped CV around the inner search |
| **hold-out test** | The participants that the selection never sees |
| **bundle** | `model.joblib`: pipeline, features, classes and feature set |
| **ECE** | Expected calibration error of the top predicted group |
| **macro-F1** | Mean F1 over the groups that occur in the test |

---

## 20. License

[MIT](LICENSE) © 2026 Krishna Annavaram
