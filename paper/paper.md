---
title: 'splitsmith: A Python package for ML dataset splitting with built-in leakage detection'
tags:
  - Python
  - machine learning
  - data splitting
  - data leakage
  - cross-validation
authors:
  - name: Shikhar Shukla
    orcid: 0009-0003-8941-3396
    corresponding: true
    affiliation: 1
affiliations:
  - name: Independent Researcher, USA
    index: 1
date: 4 March 2026
bibliography: paper.bib
---

# Summary

`splitsmith` is a Python library that combines dataset splitting and leakage
auditing into a single workflow for machine learning practitioners. It provides
train/validation/test splitting and k-fold cross-validation with four
strategies - random, group-aware, time-aware, and combined group-time, while
automatically checking each split for common data leakage patterns. The audit
system detects index overlap, duplicate rows across splits, entity (group)
leakage, and temporal leakage, returning structured reports with severity levels
and evidence. Results can be exported as JSON for experiment tracking or as
self-contained HTML reports for human review. A command-line interface enables
integration into CI/CD pipelines with pass/fail exit codes, allowing teams to
enforce leakage-free data handling as part of their automated workflows.

# Statement of need

Data leakage, where information from outside the training set improperly
influences model building, is one of the most pervasive and damaging pitfalls in
applied machine learning [@kaufman2012leakage; @kapoor2023leakage]. Leakage
inflates evaluation metrics during development but causes models to
underperform on truly unseen data in production. Common sources include
duplicate records appearing across splits, the same entity (e.g., a patient or
user) contributing rows to both training and evaluation sets, and future
observations leaking into historical training data in time-series contexts
[@arp2022dos].

Despite the severity of this problem, the standard tools used by most
practitioners do not check for leakage after splitting. Scikit-learn
[@pedregosa2011scikit] provides `train_test_split`, `GroupKFold`, and
`TimeSeriesSplit`, but none of these functions verify that the resulting split
is actually free of leakage, they only attempt to prevent it through their
splitting logic. If a user passes incorrect group labels or has duplicate rows,
the leakage goes undetected. Similarly, Pandas [@mckinney2010data] offers no
built-in splitting utilities at all.

`splitsmith` addresses this gap by coupling splitting with post-hoc auditing.
After every split, users can run an audit that checks for four categories of
leakage and returns a structured report. This "split-then-verify" workflow
catches errors that splitting logic alone cannot prevent, such as duplicate
rows that create cross-split leakage regardless of the splitting strategy used.
The library targets ML engineers, data scientists, and researchers who need
confidence that their evaluation methodology is sound, particularly in regulated
domains like healthcare and finance where leaky evaluations can have serious
consequences.

# State of the field

Several Python packages address aspects of dataset splitting for machine
learning. Scikit-learn [@pedregosa2011scikit] is the dominant library, offering
`train_test_split` for random splitting with optional stratification,
`GroupKFold` for group-aware cross-validation, and `TimeSeriesSplit` for
temporal forward-chaining. However, scikit-learn treats splitting and
validation as separate concerns; it provides no mechanism to audit a completed
split for leakage. Users must manually verify that groups do not span splits or
that temporal ordering is maintained.

`cleanlab` [@northcutt2021confident] focuses on label quality rather than split
integrity, identifying mislabeled examples rather than leakage across splits.
`deepchecks` [@deepchecks2022] includes a "train-test validation" suite that
checks for distribution drift and feature leakage between pre-defined splits,
but it does not perform the splitting itself and does not check for entity or
temporal leakage patterns. `great_expectations` [@greatexpectations2019]
validates data quality through assertions but does not address split-level
concerns.

`splitsmith` was built as a standalone package rather than a contribution to
scikit-learn for three reasons. First, the audit system requires structured
report objects (`LeakageReport`, `Finding`) with severity levels and evidence
payloads, a pattern that does not fit scikit-learn's estimator/transformer API.
Second, `splitsmith` intentionally couples splitting with auditing in a single
workflow, which is a design philosophy that differs from scikit-learn's modular
separation of concerns. Third, the export and CLI features (JSON/HTML reports,
CI exit codes) serve a production-engineering use case that extends beyond
scikit-learn's scope as a modeling library. The unique contribution of
`splitsmith` is not any individual splitting algorithm, but the integration of
splitting, auditing, reporting, and CI tooling into a unified, lightweight
package.

# Software design

`splitsmith` is organized around four core modules: `split`, `cv`, `audit`,
and `export`/`report`. The central design decision was to separate the *result*
of a split from the *evaluation* of that split, connected through shared data
structures.

The `types` module defines dataclasses that serve as the common currency across
the package. `SplitResult` holds train, validation, and test index arrays along
with strategy-specific metadata. `FoldResult` and `CVResult` extend this
pattern for cross-validation. `LeakageReport` aggregates `Finding` objects,
each with an identifier, severity level (info, warn, error), human-readable
details, and a structured evidence dictionary. The `.ok` property provides a
single boolean gate for CI integration.

The `split` module implements four strategies through a single entry point with
a `strategy` parameter. Internally, each strategy is a private function
(`_random_split`, `_group_split`, `_time_split`, `_group_time_split`) that
returns a `SplitResult`. The random strategy supports automatic stratification
detection based on target cardinality. The group strategy shuffles unique group
values and partitions them, then maps back to row indices. The time strategy
sorts by timestamp and slices chronologically. The group-time strategy sorts
groups by their maximum timestamp before partitioning, preventing both entity
and temporal leakage simultaneously.

The `cv` module mirrors these strategies for k-fold cross-validation, with
time-aware strategies using forward-chaining (expanding training window) rather
than standard k-fold rotation, reflecting best practices for temporal data
[@bergmeir2012use].

The `audit` module runs four checks against any `SplitResult`: index overlap
(set intersection), duplicate rows (hash-based detection across splits), group
leakage (shared entity values), and time leakage (train timestamps exceeding
validation/test boundaries). Each check produces findings independently,
allowing partial auditing when group or time columns are unavailable.
`audit_cv` applies the same checks to every fold of a `CVResult`.

Export functions serialize all result types to JSON, handling NumPy type
conversion automatically. HTML reports are fully self-contained (inline CSS, no
external dependencies) with color-coded severity badges. The CLI module
(`__main__.py`) wraps the full workflow, read CSV, split, audit, export, with
exit codes (0=pass, 1=leakage, 2=error) suitable for CI pipeline integration.

This architecture was chosen to keep the package lightweight (zero dependencies
beyond NumPy and Pandas) while remaining extensible. New splitting strategies
require only a private function returning `SplitResult`; new audit checks
require only a function appending `Finding` objects to a `LeakageReport`.

# Research impact statement

`splitsmith` is a newly released package and does not yet have citations in
peer-reviewed literature. To demonstrate its utility and correctness, the
package includes 136 automated tests covering all splitting strategies, audit
checks, export formats, and CLI behavior, verified across Python 3.10-3.13 on
both Linux and Windows through continuous integration.

To establish credible near-term significance, we provide a reproducible
benchmark demonstrating leakage detection on synthetic data with known
ground-truth leakage patterns. In the included demonstration notebook, a random
split of a 500-row dataset with 50 users produces group leakage across all
three split pairs (40 users in train∩val, 33 in train∩test, 27 in val∩test),
which `splitsmith` detects automatically. Switching to `strategy="group"`
eliminates all leakage, confirmed by the audit returning zero errors. This
mirrors real-world scenarios documented by @kapoor2023leakage, who found that
data leakage affected the conclusions of 329 papers across 17 scientific
fields.

The package is designed for community adoption: it is pip-installable from
GitHub, includes comprehensive documentation, follows PEP 561 for type checker
compatibility, and provides contributor guidelines.

# AI usage disclosure

Claude (Anthropic) was used as a frequent support tool during this project. The author defined the system architecture, made the key design choices, and produced phased specifications, then wrote the initial implementation for all modules and associated tests from those plans. Claude contributed selectively, primarily by suggesting improvements to error handling and helping draft documentation and sections of the manuscript. All code and text were executed, checked, and validated by the author, who assumes full responsibility for the completed work.

# Acknowledgements

The author thanks the developers of NumPy, Pandas, and scikit-learn for the
foundational tools upon which `splitsmith` builds.

# References
