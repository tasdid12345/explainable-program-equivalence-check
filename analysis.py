"""Does behavioural evidence (running both programs) beat a surface-feature judge
of semantic equivalence for small generated Python functions?

Pipeline: seeded generator -> label each pair by output agreement on a dense grid
-> stratified train/test split (fixed seed) -> CV-only tuning on train -> one test evaluation.
"""
import json
import os
import random
import re
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED, SEEDS = 42, [42, 43, 44]
N_PAIRS = 4000
QUESTION = ("Does adding behavioural evidence from running both programs on test inputs let a "
            "gradient-boosted classifier beat a surface-feature baseline at judging whether two "
            "small programs are semantically equivalent?")

# Operators, comparisons and their negations; variable names for the local "z" (renamed by rewrites).
BIN = ["+", "-", "*"]
CMP = ["<", "<=", ">", ">=", "==", "!="]
NEG = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "!=": "=="}
VARS = ["w", "t", "acc", "res", "q"]
XY, XYZ = ["x", "y"], ["x", "y", "z"]
GRID = [(x, y) for x in range(-6, 7) for y in range(-6, 7)]          # dense grid -> label
BEH = np.random.default_rng(SEED).integers(-50, 51, size=(64, 2)).tolist()  # separate behaviour inputs
TOK = re.compile(r"\w+|[^\w\s]")
SURF_NAMES = ["tok_multiset_jaccard", "tok_set_jaccard", "identifier_jaccard", "number_jaccard",
              "operator_jaccard", "len_min", "len_max", "len_absdiff", "len_ratio",
              "tokcount_absdiff", "if_count_absdiff", "identical_text"]
BEH_NAMES = ["agree_rate", "mean_abs_diff", "max_abs_diff", "std_diff", "frac_a_gt_b",
             "frac_b_gt_a", "std_out_a", "std_out_b", "same_output_set"]
r = random.Random(SEED)  # the only generator used for programs


# ---------- program generator: v = a; if cond: v = b else: v = c; return ret ----------
def rand_expr(depth, names):
    """Random arithmetic expression as a nested tuple tree."""
    if depth == 0 or r.random() < 0.3:
        return ("var", r.choice(names)) if r.random() < 0.6 else ("const", r.randint(-5, 9))
    if r.random() < 0.15:
        return ("mod", rand_expr(depth - 1, names), r.randint(2, 7))
    return (r.choice(BIN), rand_expr(depth - 1, names), rand_expr(depth - 1, names))


def rand_prog():
    return {"v": "z", "a": rand_expr(2, XY),
            "cond": (r.choice(CMP), rand_expr(1, XY), rand_expr(1, XY)),
            "b": rand_expr(2, XYZ), "c": rand_expr(2, XYZ), "ret": rand_expr(2, XYZ)}


def show(e, v):
    """Render an expression tuple as Python source; 'z' is mapped to the local name v."""
    t = e[0]
    if t == "var":
        return v if e[1] == "z" else e[1]
    if t == "const":
        return str(e[1]) if e[1] >= 0 else f"({e[1]})"
    if t == "mod":
        return f"({show(e[1], v)} % {e[2]})"
    return f"({show(e[1], v)} {t} {show(e[2], v)})"


def render(p, v, swap=False):
    """Source text. swap=True emits negated condition with swapped arms (same behaviour)."""
    def S(e):
        return show(e, v)
    op, l, rr = p["cond"]
    then, other = (p["c"], p["b"]) if swap else (p["b"], p["c"])
    cop = NEG[op] if swap else op
    return (f"def f(x, y):\n    {v} = {S(p['a'])}\n    if {S(l)} {cop} {S(rr)}:\n"
            f"        {v} = {S(then)}\n    else:\n        {v} = {S(other)}\n"
            f"    return {S(p['ret'])}\n")


def rewrite(e):
    """Semantics-preserving rewrite: commutative swap of + and *, and +0 added to leaves."""
    t = e[0]
    if t in ("var", "const"):
        return ("+", e, ("const", 0)) if r.random() < 0.2 else e
    if t == "mod":
        return ("mod", rewrite(e[1]), e[2])
    l, rr = rewrite(e[1]), rewrite(e[2])
    if t in ("+", "*") and r.random() < 0.5:
        l, rr = rr, l
    return (t, l, rr)


def rewrite_prog(p):
    """Equivalent variant: renamed local, rewritten expressions, random negated-swap flag."""
    op, l, rr = p["cond"]
    q = {"v": r.choice(VARS), "a": rewrite(p["a"]), "b": rewrite(p["b"]), "c": rewrite(p["c"]),
         "cond": (op, rewrite(l), rewrite(rr)), "ret": rewrite(p["ret"])}
    return q, r.random() < 0.5


def mutate_expr(e):
    """One behaviour-changing edit somewhere in an expression tree."""
    t = e[0]
    if t == "const":
        return ("const", e[1] + r.choice([-2, -1, 1, 2]))
    if t == "var":
        return ("const", r.randint(-3, 3))
    if t == "mod":
        return ("mod", e[1], e[2] + 1) if r.random() < 0.5 else ("mod", mutate_expr(e[1]), e[2])
    if r.random() < 0.25:
        return (r.choice([o for o in BIN if o != t]), e[1], e[2])
    if r.random() < 0.5:
        return (t, mutate_expr(e[1]), e[2])
    return (t, e[1], mutate_expr(e[2]))


def mutate_prog(p):
    """Inequivalence-prone variant: flip comparison, swap arms without negation, or edit an expression."""
    q = dict(p)
    k = r.random()
    if k < 0.2:
        q["cond"] = (r.choice([o for o in CMP if o != p["cond"][0]]), p["cond"][1], p["cond"][2])
    elif k < 0.3:
        q["b"], q["c"] = p["c"], p["b"]
    else:
        key = r.choice(["a", "b", "c", "ret"])
        q[key] = mutate_expr(p[key])
    return q


def make_pairs(n):
    """Pairs of (source_a, source_b); about half are rewrites, most of the rest are mutations."""
    pairs = []
    for _ in range(n):
        p = rand_prog()
        u = r.random()
        if u < 0.45:
            q, swap = rewrite_prog(p)
            sb = render(q, q["v"], swap)
        elif u < 0.85:
            sb = render(mutate_prog(p), "z")
        else:
            sb = render(rand_prog(), "z")
        pairs.append((render(p, "z"), sb))
    return pairs


# ---------- features ----------
def run_src(src):
    """Compile generated source (internal generator output only) and return f."""
    ns = {}
    exec(src, ns)
    return ns["f"]


def jac(a, b):
    return len(a & b) / max(1, len(a | b))


def surface_feats(sa, sb):
    """Baseline features: token and identifier overlap plus length and token-count differences."""
    ta, tb = TOK.findall(sa), TOK.findall(sb)
    ca, cb = Counter(ta), Counter(tb)
    ia, ib = {t for t in ta if t.isidentifier()}, {t for t in tb if t.isidentifier()}
    na, nb = {t for t in ta if t.isdigit()}, {t for t in tb if t.isdigit()}
    oa = {t for t in ta if not t.isidentifier() and not t.isdigit()}
    ob = {t for t in tb if not t.isidentifier() and not t.isdigit()}
    la, lb = len(sa), len(sb)
    return [sum((ca & cb).values()) / max(1, sum((ca | cb).values())), jac(set(ta), set(tb)),
            jac(ia, ib), jac(na, nb), jac(oa, ob), min(la, lb), max(la, lb), abs(la - lb),
            min(la, lb) / max(la, lb), abs(len(ta) - len(tb)),
            abs(ta.count("if") - tb.count("if")), float(sa == sb)]


def featurize(sa, sb):
    """Label (1 = inequivalent, from the dense grid), surface features, behavioural features."""
    fa, fb = run_src(sa), run_src(sb)
    label = int([fa(x, y) for x, y in GRID] != [fb(x, y) for x, y in GRID])
    da = np.array([fa(x, y) for x, y in BEH], float)
    db = np.array([fb(x, y) for x, y in BEH], float)
    d = db - da
    beh = [np.mean(d == 0), np.mean(np.abs(d)), np.max(np.abs(d)), np.std(d), np.mean(d > 0),
           np.mean(d < 0), np.std(da), np.std(db), float(set(da) == set(db))]
    return label, surface_feats(sa, sb), beh


# ---------- models and evaluation ----------
def run_model(name, base, grid, Xtr, ytr, Xte, yte):
    """Tune on train only (3-fold CV), then fit with each seed and score once on test."""
    gs = GridSearchCV(base, grid, cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
                      scoring="accuracy").fit(Xtr, ytr)
    print(f"[{name}] CV-selected {gs.best_params_} (train CV acc {gs.best_score_:.3f})")
    res, models = [], []
    for s in SEEDS:
        m = clone(base).set_params(**gs.best_params_)
        m.set_params(**{k: s for k in m.get_params() if k.endswith("random_state")})
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        res.append({"acc": accuracy_score(yte, p), "f1": f1_score(yte, p, pos_label=1),
                    "fer": float(np.mean(p[yte == 1] == 0))})  # inequivalent judged equivalent
        print(f"[{name}] seed {s}: test acc {res[-1]['acc']:.4f}")
        models.append(m)
    return res, models


def agg(res, key):
    v = np.array([d[key] for d in res], float)
    return float(v.mean()), float(v.std())


def main():
    print("Generating program pairs...")
    rows = [featurize(a, b) for a, b in make_pairs(N_PAIRS)]
    y = np.array([t[0] for t in rows])
    S = np.array([t[1] for t in rows], float)
    B = np.array([t[2] for t in rows], float)
    X = np.hstack([S, B])
    print(f"Pairs: {len(y)}, inequivalent share: {y.mean():.3f}")
    if min(y.mean(), 1 - y.mean()) < 0.05:
        print("ERROR: degenerate label balance"); sys.exit(1)

    tr, te = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y, random_state=SEED)
    print(f"Split: train {len(tr)}, test {len(te)}")

    base = make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED))
    b_res, _ = run_model("baseline", base, {"logisticregression__C": [0.1, 1, 10]},
                         S[tr], y[tr], S[te], y[te])
    imp = LGBMClassifier(n_estimators=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                         class_weight="balanced", random_state=SEED, verbose=-1)
    i_res, i_models = run_model("improved", imp, {"num_leaves": [15, 31], "learning_rate": [0.05, 0.1]},
                                X[tr], y[tr], X[te], y[te])

    b_acc, b_sd = agg(b_res, "acc")
    i_acc, i_sd = agg(i_res, "acc")
    if not (np.isfinite(b_acc) and np.isfinite(i_acc)):
        print("ERROR: non-finite metrics"); sys.exit(1)

    # Gain-based importance of the improved model (seed 42 fit) for explaining verdicts.
    names = SURF_NAMES + BEH_NAMES
    gain = i_models[0].booster_.feature_importance("gain")
    order = np.argsort(gain)[::-1]
    print("Top features (gain):", [names[i] for i in order[:5]])

    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.bar(["baseline (surface LR)", "improved (LGBM + behaviour)"], [b_acc, i_acc],
            yerr=[b_sd, i_sd], color=["grey", "steelblue"], capsize=6)
    plt.ylabel("test accuracy"); plt.tight_layout()
    plt.savefig("figures/accuracy_comparison.png", dpi=120); plt.close()
    plt.figure(figsize=(6, 4))
    plt.barh([names[i] for i in order[:10]][::-1], gain[order[:10]][::-1], color="steelblue")
    plt.xlabel("gain importance"); plt.tight_layout()
    plt.savefig("figures/feature_importance.png", dpi=120); plt.close()

    out = {
        "question": QUESTION,
        "method": ("Seeded generator of small arithmetic/branching Python function pairs; label = outputs "
                   "differ on a dense grid. Stratified train/test split with fixed seed. Baseline and improved "
                   "models are tuned by 3-fold CV on train only and scored once on test with seeds 42-44."),
        "dataset": {"name": "Synthetic arithmetic-and-branching Python function pairs (seeded generator)",
                    "source_url": "https://numpy.org", "n_rows": int(len(y)), "n_train": int(len(tr)),
                    "n_test": int(len(te)), "same_as_paper": False, "baseline_is_papers_method": False},
        "metric_name": "accuracy (held-out test split)",
        "higher_is_better": True,
        "baseline": ("Logistic regression (standardised, class-balanced) on surface token-overlap and "
                     "length features of the two sources; no execution."),
        "improved": ("LightGBM classifier (class-balanced) on the same surface features plus behavioural "
                     "features from executing both programs on 64 separate random inputs (agreement rate, "
                     "output-difference statistics); gain importance reported."),
        "metrics": {
            "baseline_score": b_acc, "improved_score": i_acc, "improvement": i_acc - b_acc,
            "baseline_score_std": b_sd, "improved_score_std": i_sd,
            "baseline_f1": agg(b_res, "f1")[0], "improved_f1": agg(i_res, "f1")[0],
            "baseline_f1_std": agg(b_res, "f1")[1], "improved_f1_std": agg(i_res, "f1")[1],
            "baseline_false_equiv_rate": agg(b_res, "fer")[0],
            "improved_false_equiv_rate": agg(i_res, "fer")[0],
            "test_inequivalent_share": float(y[te].mean()),
        },
        "notes": ("Labels are grid-based (outputs agree on a fixed 13x13 grid), so pairs that differ only "
                  "outside the grid are labelled equivalent; this is a small stand-in, not the paper's "
                  "LLM benchmark. Both models use class-balanced weights. Logistic regression is "
                  "deterministic, so its std across seeds is 0. No downloads are made, so no retry logic "
                  "is needed. Generated code is internal generator output only, executed via exec. "
                  "Test set used once; all tuning uses CV on train."),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Baseline acc {b_acc:.4f} +/- {b_sd:.4f} | Improved acc {i_acc:.4f} +/- {i_sd:.4f}")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
