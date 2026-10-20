# Behavioural evidence for judging program equivalence

Large language models are increasingly asked to judge whether two programs are semantically equivalent, and the study "Understanding code semantics: a benchmark study of LLMs" evaluates how well they do this. Running LLM judges at scale is costly and hard to reproduce on a CPU, so this experiment asks a cheaper, reproducible question about the same task.

**Question:** Does adding behavioural evidence from running both programs on test inputs let a gradient-boosted classifier beat a surface-feature baseline at judging whether two small programs are semantically equivalent?

**Dataset:** A synthetic set of small arithmetic-and-branching Python function pairs, generated with a seeded generator. Each pair is labelled equivalent or inequivalent by comparing outputs on a fixed dense input grid. This is not the paper's own dataset (`same_as_paper: false`).

**Baseline:** Logistic regression on surface token-overlap and length features of the two sources. This is a standard stand-in for an LLM judge, not the paper's method.

**Improved model:** LightGBM with class-balanced weights, using the same surface features plus behavioural features from executing both programs on separate random inputs (agreement rate and output-difference statistics). It also reports per-feature importance.

Both models use the same data, stratified train/test split (fixed seed), CV-only tuning on the training split, and test evaluation.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

## Results

Results are written to `results.json` (charts are saved in `figures/`).

## Reference

This project is motivated by the following paper:

> C Laneve, A Spanò, D Ressi, S Rossi (2026). *Understanding code semantics: a benchmark study of LLMs*. International Journal on. https://link.springer.com/article/10.1007/s10009-026-00842-4

Paper author: Prof. Michele Bugliesi, Ca' Foscari University of Venice (bugliesi@unive.it)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
