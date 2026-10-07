# The writing standard: ASD-STE100 Simplified Technical English

Use these rules for every README and for `docs/ste-style-guide.md` in each repository. Copy this file
into the repository as `docs/ste-style-guide.md` and add a **project vocabulary** section (Section 3)
with the technical names and technical verbs of that project.

## 1. The writing rules

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, `test` is a noun or a verb, `check` is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: `prepare`, `do`, `find`, `get`, `make`.
4. Do not use an `-ing` form as a noun or an adjective (`the running job`, `after indexing`).
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and `can` for a possibility.
8. Keep the articles `a`, `an` and `the` in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: `If the index is stale, build it again.`
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase (`The cost model`) or an imperative (`Run the demo`).
   Do not start a heading with an `-ing` form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or `check that` |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

This section gives the technical names and the technical verbs of dementia-typer. The README uses each term with only this meaning.

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **participant** | One person in OASIS-3, identified by `OASISID` | subject (except in code names), patient, case |
| **visit** | One clinical visit of a participant, identified by `OASISID` and `days_to_visit` | session (for a clinical visit), record, row |
| **MR session** | One MRI scan with FreeSurfer volumes, identified by `OASISID` and `days_to_scan` | scan session, image |
| **join window** | The maximum distance in days between a visit and its MR session | tolerance, match range |
| **group** | One of the 5 diagnosis classes in `CLASSES` | class label, category, subtype (alone) |
| **dx text** | The `dx1` diagnosis text of a visit | diagnosis code, label text |
| **rule** | One ordered pattern in `labels.RULES` that maps a dx text to a group | substring match, heuristic |
| **feature set** | `A`, `B` or `C`, the list of feature columns of an experiment | feature group, view |
| **leakage ceiling** | The result of feature set `C`, which contains CDR scores | upper bound, best case |
| **CDR** | Clinical Dementia Rating: the six boxes, `CDRSUM` and `CDRTOT` | dementia score |
| **MMSE** | Mini-Mental State Examination score, 0 to 30 | cognitive score (alone) |
| **candidate** | One model and selector pair in the inner search | configuration, option |
| **inner search** | The grouped grid search on the training participants | tuning, validation |
| **hold-out test** | The participants that the selection never sees | test set (alone), validation set |
| **pipeline** | The scikit-learn `Pipeline` of imputer, scaler, selector and model | workflow, chain |
| **bundle** | The `model.joblib` file with the pipeline, features, classes and feature set | model file, pickle |
| **baseline** | The `majority` model | dummy, naive model |
| **model card** | The Markdown file with the intended use, metrics and limits of a bundle | report, datasheet |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **map** | Change a dx text into a group with the rules |
| **join** | Attach the nearest MR session in the join window to a visit |
| **validate** | Check the tables, the join and the rules |
| **select** | Choose the candidate with the highest inner macro-F1 |
| **evaluate** | Measure the selected pipeline once on the hold-out test |
| **predict** | Give group probabilities for one validated visit |
| **resample** | Draw participants with replacement for a confidence interval |
