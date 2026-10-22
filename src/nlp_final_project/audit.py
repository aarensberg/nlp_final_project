"""Bias and failure-mode audit of the classifier (3 variants) and of the retriever.

The shipped variant is the combination (`ensemble`); BERT alone and TF-IDF are
reported next to it. Each variant is always used with ITS OWN validation-chosen
abstention threshold (reports/inference_config.json).

1. Slices of the test set: contract family, pronouns, jurisdiction, clause
   length, class frequency. Groups are KEYWORD PROXIES computed from the clause
   text (see slices.py), not metadata and not certified personal attributes.
   Raw gaps are confounded by each group's clause-type mix, so we also report the
   accuracy expected from that mix; the adjusted gap is descriptive only. Small
   groups (wide intervals) support no conclusion about fairness or discrimination.
2. Counterfactual edits that should not change the clause TYPE (perturb.py).
   Some edits (jurisdiction, currency) DO change legal content: a stable type
   does not mean the legal effect is preserved.
3. Out-of-scope inputs (types LEDGAR lacks, French/German text, non-legal text).
   A confidence threshold does not guarantee they are caught.
4. Retrieval: quality per contract family, cross-family exposure, stability
   under the gender swap, pronoun make-up of the clause bank.

Run: uv run nlp-final-project audit
"""

from __future__ import annotations

import numpy as np

from . import plotting
from .config import FIGURES, REPORTS, SEED, save_json
from .data import frequency_buckets, label_names, split_arrays
from .inference import VARIANTS, Predictor, split_logits
from .metrics import answers
from .perturb import PERTURBATIONS, gender_swap, has_gender
from .slices import CONTRACT_FAMILIES, compute_slices

VLABEL = {"ensemble": "Combination", "bert": "Legal-BERT alone", "tfidf": "TF-IDF"}
VCOLOR = {"ensemble": plotting.SERIES[0], "bert": plotting.SERIES[1], "tfidf": plotting.SERIES[2]}
SMALL_GROUP = 100  # below this, intervals are too wide to read anything into a gap

# Hand-written inputs the system should NOT label confidently.
OUT_OF_SCOPE = [
    ("GDPR data processing", "The Processor shall process Personal Data only on documented instructions from the Controller, including with regard to transfers of personal data to a third country, in accordance with Article 28 of Regulation (EU) 2016/679."),
    ("Force majeure", "Neither Party shall be liable for any failure or delay in performance caused by events beyond its reasonable control, including acts of God, pandemic, war, terrorism, strikes or governmental action."),
    ("Non-compete", "During the term and for twelve months thereafter, the Seller shall not, directly or indirectly, engage in any business that competes with the Business in the European Economic Area."),
    ("French governing law (FR)", "Le présent contrat est régi par le droit français. Tout litige relatif à son interprétation ou à son exécution relèvera de la compétence exclusive du Tribunal de commerce de Paris."),
    ("German confidentiality (DE)", "Die Parteien verpflichten sich, alle im Rahmen dieses Vertrages erhaltenen vertraulichen Informationen geheim zu halten und nicht an Dritte weiterzugeben."),
    ("French termination (FR)", "Chaque partie pourra résilier le présent contrat moyennant un préavis de trente (30) jours notifié par lettre recommandée avec accusé de réception."),
    ("E-mail, not a clause", "Hi team, please find attached the signed version of the supplier agreement. Let me know if you need anything else before Friday."),
    ("Random text", "Preheat the oven to 180 degrees, mix the flour with the eggs and bake for twenty-five minutes."),
]
OOS_NOTE = ("None of these inputs has a correct LEDGAR label (or is not an English contract clause). "
            "Any answered prediction is a confident error. A confidence threshold reduces but does not "
            "guarantee the detection of out-of-scope inputs.")


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


# ------------------------------------------------------------------ slices ----
def slice_report(y, probs: dict, thresholds: dict, slices: dict[str, np.ndarray]) -> dict:
    out = {}
    recall = {}
    for v in VARIANTS:
        pred = probs[v].argmax(1)
        recall[v] = np.array([((pred == c) & (y == c)).sum() / max((y == c).sum(), 1)
                              for c in range(probs[v].shape[1])])
    for name, arr in slices.items():
        rows = []
        for g in sorted(set(arr.tolist())):
            m = arr == g
            n = int(m.sum())
            row = {"group": g, "n": n, "small_group": n < SMALL_GROUP}
            for v in VARIANTS:
                p = probs[v][m]
                pred = p.argmax(1)
                k = int((pred == y[m]).sum())
                ans = answers(p.max(1), thresholds[v])
                expected = float(recall[v][y[m]].mean())
                ci = wilson(k, n)
                row[v] = {
                    "accuracy": k / n, "accuracy_95ci": ci,
                    "expected_from_clause_type_mix": expected,
                    "mix_adjusted_gap_points": (k / n - expected) * 100,
                    "below_mix_expectation_beyond_95ci": bool(ci[1] < expected),
                    "coverage": float(ans.mean()),
                    "correct_rate_among_handled": float((pred[ans] == y[m][ans]).mean()) if ans.any() else None,
                    "n_handled": int(ans.sum()),
                }
            rows.append(row)
        out[name] = rows
    return out


# ---------------------------------------------------------- counterfactuals ----
def counterfactuals(P: Predictor, X: list[str], y: np.ndarray, names: list[str]) -> tuple[list, dict, dict]:
    rows, flips_ex, samples = [], {}, {}
    rng = np.random.default_rng(SEED)
    for pname, (fn, applies, meaning) in PERTURBATIONS.items():
        idx = [i for i, t in enumerate(X) if applies(t)]
        if pname.startswith("3%"):
            idx = sorted(rng.choice(len(X), 1000, replace=False).tolist())
        pairs = [(i, X[i], fn(X[i])) for i in idx]
        pairs = [p for p in pairs if p[1] != p[2]]
        if not pairs:
            continue
        ii = np.array([p[0] for p in pairs])
        before = P.probs([p[1] for p in pairs])
        after = P.probs([p[2] for p in pairs])
        yy = y[ii]
        row = {"perturbation": pname, "meaning": meaning, "n": len(pairs)}
        for v in VARIANTS:
            b, a = before[v], after[v]
            flips = b.argmax(1) != a.argmax(1)
            thr = P.threshold(v)
            abst_change = answers(b.max(1), thr) != answers(a.max(1), thr)
            row[v] = {
                "type_flip_rate": float(flips.mean()), "type_flip_rate_95ci": wilson(int(flips.sum()), len(pairs)),
                "n_flips": int(flips.sum()),
                "accuracy_before": float((b.argmax(1) == yy).mean()),
                "accuracy_after": float((a.argmax(1) == yy).mean()),
                "abstention_decision_changed_rate": float(abst_change.mean()),
                "mean_abs_change_true_label_score": float(np.abs(b[np.arange(len(yy)), yy]
                                                                 - a[np.arange(len(yy)), yy]).mean()),
            }
            if v == "ensemble":
                flips_ex[pname] = [{"before": pairs[j][1][:300], "after": pairs[j][2][:300],
                                    "pred_before": names[b[j].argmax()], "pred_after": names[a[j].argmax()],
                                    "true": names[yy[j]]} for j in np.where(flips)[0][:4]]
        rows.append(row)
        # Random before/after samples, kept for manual review of the edit rules.
        pick = rng.choice(len(pairs), min(4, len(pairs)), replace=False)
        samples[pname] = [{"before": _diff_window(pairs[j][1], pairs[j][2])[0],
                           "after": _diff_window(pairs[j][1], pairs[j][2])[1]} for j in pick]
    return rows, flips_ex, samples


def _diff_window(a: str, b: str, ctx: int = 90) -> tuple[str, str]:
    """Short excerpt around the first difference, so reviewers see the edit."""
    i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    s = max(0, i - ctx)
    return a[s:i + ctx], b[s:i + ctx]


# ----------------------------------------------------------------- retrieval ----
def retrieval_audit(Xte, yte, slices) -> dict:
    from .retrieval import ClauseIndex, norm_hash

    idx = ClauseIndex.load()
    rng = np.random.default_rng(SEED)
    fam = slices["contract_family"]
    bank_slices = compute_slices(idx.texts)
    bank_fam = bank_slices["contract_family"]
    res: dict = {"protocol": "test clause as query, hybrid (dense+BM25) top-10, exact duplicates of the "
                             "query removed, relevant = same LEDGAR label; <=150 queries per family"}

    sel = []
    for g in sorted(set(fam.tolist())):
        pool = np.where(fam == g)[0]
        sel += rng.choice(pool, min(150, len(pool)), replace=False).tolist()
    qemb = idx.embed([Xte[i] for i in sel], is_question=False)
    per_fam: dict[str, list[float]] = {}
    exposure: dict[str, int] = {}
    for i, e in zip(sel, qemb):
        s, ranked = idx.scored("hybrid", idx.dense_scores(e), idx.bm25_scores(Xte[i]))
        top = idx.top(s, ranked, 10, exclude_hash=norm_hash(Xte[i]))
        per_fam.setdefault(fam[i], []).append(float((idx.labels[top] == yte[i]).sum() / 10))
        for j in top:
            if bank_fam[j] != fam[i]:
                exposure[bank_fam[j]] = exposure.get(bank_fam[j], 0) + 1
    res["hybrid_p@10_by_contract_family"] = {
        k: {"p@10": float(np.mean(v)), "n_queries": len(v), "sd": float(np.std(v))} for k, v in per_fam.items()}
    bank_share = {g: float((bank_fam == g).mean()) for g in sorted(set(bank_fam.tolist()))}
    tot = sum(exposure.values()) or 1
    res["cross_family_results_share_vs_bank_share"] = {
        g: {"share_of_cross_family_results": exposure.get(g, 0) / tot, "share_of_bank": bank_share[g]}
        for g in bank_share}

    gi = [i for i, t in enumerate(Xte) if has_gender(t)][:300]
    a = idx.embed([Xte[i] for i in gi], is_question=False)
    b = idx.embed([gender_swap(Xte[i]) for i in gi], is_question=False)
    jac, same_label_rate = [], []
    for i, ea, eb in zip(gi, a, b):
        ta = np.array(idx.top(*idx.scored("hybrid", idx.dense_scores(ea), idx.bm25_scores(Xte[i])), 10))
        tb = np.array(idx.top(*idx.scored("hybrid", idx.dense_scores(eb),
                                          idx.bm25_scores(gender_swap(Xte[i]))), 10))
        jac.append(len(set(ta) & set(tb)) / len(set(ta) | set(tb)))
        same_label_rate.append(((idx.labels[ta] == yte[i]).sum() / 10, (idx.labels[tb] == yte[i]).sum() / 10))
    sl = np.array(same_label_rate)
    res["gender_swap_top10"] = {
        "n_queries": len(jac), "mean_jaccard": float(np.mean(jac)),
        "share_identical_sets": float(np.mean(np.array(jac) == 1)),
        "p@10_original": float(sl[:, 0].mean()), "p@10_swapped": float(sl[:, 1].mean()),
        "reading": "results change (different precedents shown) even when relevance stays similar"}
    pr = bank_slices["pronouns"]
    res["bank_pronoun_share"] = {g: float((pr == g).mean()) for g in sorted(set(pr.tolist()))}
    return res


# ----------------------------------------------------------------------- run ----
def run() -> dict:
    names = label_names()
    Xtr, ytr = split_arrays("train")
    Xte, yte = split_arrays("test")
    P = Predictor()
    lb, y_b = split_logits("test", "bert")
    lt, _ = split_logits("test", "tfidf")
    assert np.array_equal(y_b, yte)
    probs = P.probs_from_logits(lb, lt)
    thresholds = {v: P.threshold(v) for v in VARIANTS}

    slices = compute_slices(Xte)
    buckets = frequency_buckets(ytr, len(names))
    slices["class_frequency"] = np.array([buckets[c] for c in yte])

    out: dict = {
        "primary_variant": "ensemble",
        "thresholds_used": thresholds,
        "caveats": [
            "Groups are keyword proxies inferred from clause text (slices.py), not metadata, "
            "not certified personal attributes; a clause can be mis-assigned.",
            f"Groups with n < {SMALL_GROUP} have wide intervals; no conclusion on fairness or discrimination "
            "can be drawn from them.",
            "The mix-adjusted gap compares a group with the accuracy its clause-type mix would predict; "
            "it is descriptive, not causal.",
            "Counterfactual edits test stability of the clause TYPE; jurisdiction and currency edits change "
            "the legal content, so a stable type does not imply an equivalent clause.",
        ],
        "slice_definitions": {"contract_family_keywords": CONTRACT_FAMILIES,
                              "class_frequency": "terciles of training count (head/torso/tail)",
                              "pronouns": "he/him/his/himself vs she/her/hers/herself in the clause",
                              "jurisdiction": "US state or non-US place named in the clause"},
        "slices": slice_report(yte, probs, thresholds, slices),
    }
    cf, flips_ex, samples = counterfactuals(P, Xte, yte, names)
    out["counterfactuals"] = cf
    out["counterfactual_flip_examples_ensemble"] = flips_ex
    out["counterfactual_edit_samples_for_review"] = samples

    p_oos = P.probs([t for _, t in OUT_OF_SCOPE])
    items = []
    for j, (n, t) in enumerate(OUT_OF_SCOPE):
        row = {"input": n, "text": t}
        for v in VARIANTS:
            r = p_oos[v][j]
            row[v] = {"predicted": names[r.argmax()], "confidence": float(r.max()),
                      "abstains": not bool(answers(r.max(), thresholds[v]))}
        items.append(row)
    out["out_of_scope"] = {
        "note": OOS_NOTE, "items": items,
        "confident_errors": {v: [it["input"] for it in items if not it[v]["abstains"]] for v in VARIANTS},
    }
    out["retrieval"] = retrieval_audit(Xte, yte, slices)
    save_json(out, REPORTS / "audit.json")
    _figures(out)
    return out


def _figures(out):
    plt = plotting.plt
    panels = ("contract_family", "jurisdiction", "pronouns", "length", "class_frequency")
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 4.8), sharey=True,
                             gridspec_kw={"width_ratios": [6, 3, 4, 4, 3]})
    width = 0.26
    for ax, k in zip(axes, panels):
        rows = out["slices"][k]
        x = np.arange(len(rows))
        for o, v in zip((-width, 0, width), VARIANTS):
            acc = np.array([r[v]["accuracy"] for r in rows])
            lo = acc - np.array([r[v]["accuracy_95ci"][0] for r in rows])
            hi = np.array([r[v]["accuracy_95ci"][1] for r in rows]) - acc
            ax.bar(x + o, acc, width=width * 0.92, color=VCOLOR[v], label=VLABEL[v])
            ax.errorbar(x + o, acc, yerr=[lo, hi], fmt="none", ecolor=plotting.INK_2, lw=0.8, capsize=1.5)
        ax.scatter(x - width, [r["ensemble"]["expected_from_clause_type_mix"] for r in rows], marker="_",
                   s=120, lw=2, color=plotting.INK, zorder=4, label="Combination: expected from type mix")
        ax.set_xticks(x, [f"{r['group']}\n(n={r['n']}{', small' if r['small_group'] else ''})" for r in rows],
                      fontsize=7, rotation=45, ha="right")
        ax.set_title(k.replace("_", " "), fontsize=10)
        ax.set_ylim(0.4, 1.0)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Test accuracy (95% Wilson CI)")
    h, lbl = axes[0].get_legend_handles_labels()
    fig.legend(h, lbl, loc="upper right", ncol=4, fontsize=8, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Accuracy by keyword-proxy slice (descriptive; small groups are inconclusive)",
                 x=0.01, ha="left", fontweight="semibold")
    fig.subplots_adjust(top=0.82, wspace=0.08)
    plotting.save(fig, FIGURES / "slice_accuracy.png")

    cf = out["counterfactuals"]
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(cf))
    h = 0.26
    xmax = 0
    for o, v in zip((-h, 0, h), VARIANTS):
        fr = np.array([r[v]["type_flip_rate"] for r in cf])
        lo = fr - np.array([r[v]["type_flip_rate_95ci"][0] for r in cf])
        hi = np.array([r[v]["type_flip_rate_95ci"][1] for r in cf]) - fr
        ax.barh(y + o, fr, height=h * 0.92, color=VCOLOR[v], label=VLABEL[v])
        ax.errorbar(fr, y + o, xerr=[lo, hi], fmt="none", ecolor=plotting.INK_2, lw=0.8, capsize=1.5)
        xmax = max(xmax, (fr + hi).max())
    for yi, r in zip(y, cf):
        ax.text(xmax * 1.05, yi - h, f"{r['ensemble']['type_flip_rate']:.1%} (n={r['n']})", va="center",
                fontsize=7, color=plotting.INK)
    ax.set_yticks(y, [r["perturbation"] for r in cf], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax * 1.35)
    ax.set_xlabel("Share of edited clauses whose predicted TYPE changes (95% CI)")
    ax.set_title("Counterfactual edits (label: combination flip rate)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8)
    plotting.save(fig, FIGURES / "counterfactual_flips.png")


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2)[:6000])
