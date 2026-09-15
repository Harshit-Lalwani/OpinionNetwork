# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

This is Assignment 1 ("Opinion Network Formation") for a course on Dynamical Processes and Complex Networks. It is an open-ended, team-graded assignment: use the survey dataset to construct and analyze a network of opinions, then produce an ~8-page report. There is no application code, build system, or test suite here — this repo is a data-analysis workspace centered on `EDA.ipynb` (currently empty).

## Files

- `Assignment_guidelines.md` — the assignment brief. Required report sections, in order: Team Name, GitHub link, Dataset Documentation, Pipeline Followed, Analysis and Visualizations, Results and Discussion, Individual Contribution.
- `Survey_Results_UC.csv` — the raw dataset: 96 respondents (`id. Response ID`) x 60 survey statements. Statement columns are prefixed by category:
  - `T01`–`T15`: Technology
  - `E01`–`E15`: Education
  - `S01`–`S15`: Ethics/Society
  - `V01`–`V15`: Environment
  Each cell is a 5-point Likert response: `Strongly Disagree`, `Disagree`, `Neutral`, `Agree`, `Strongly Agree` (plus occasional blank / `No Comments`). Read the CSV with `encoding='utf-8-sig'` — it has a UTF-8 BOM.
- `EDA.ipynb` — intended location for exploratory data analysis and network construction/analysis code.

## Working on this assignment

- The core task is to turn the Likert-response matrix into a network (e.g., respondents as nodes with edges from opinion similarity/distance across statements, or a bipartite respondent-statement structure) and analyze it with standard network metrics (centrality, community structure, etc.).
- No fixed pipeline or library choice is prescribed by the assignment — approach is open-ended, so don't assume a particular network-construction method without checking `EDA.ipynb`'s current contents first.
- The final deliverable is a PDF report following the exact section order in `Assignment_guidelines.md`, plus an accessible GitHub repo link and an individual contribution breakdown (grading is per-individual, not per-team).
