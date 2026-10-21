"""Bias and failure-mode audit of the classifier and the retriever.

1. Slice performance: contract family, pronouns, jurisdiction, clause length,
   class frequency. Raw gaps are confounded by the clause-type mix of each slice,
   so we also report the gap *after* adjusting for that mix.
2. Counterfactual invariance: edits that must not change the clause type
   (gender of pronouns, US -> European jurisdiction, $ -> EUR, British spelling,
   European party names, typos). We measure how often the prediction flips.
3. Out-of-scope inputs: clause types LEDGAR does not have (GDPR, force majeure),
   non-English clauses, non-legal text. Does the abstention threshold catch them?
4. Retrieval: quality per slice, exposure of each contract family in results,
   and stability of results under the gender swap.

Run: uv run nlp-final-project audit
"""

from __future__ import annotations

import re

import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import plotting
from .config import ARTIFACTS, FIGURES, REPORTS, SEED, get_device, load_json, save_json
from .data import frequency_buckets, label_names, split_arrays
from .evaluate import MODEL_DIR, model_logits
from .metrics import softmax
from .slices import compute_slices
from .train import predict_logits, tokenize

# ------------------------------------------------------ perturbations ----
_GENDER = {"he": "she", "she": "he", "him": "her", "his": "her", "her": "his", "hers": "his",
           "himself": "herself", "herself": "himself", "mr.": "ms.", "ms.": "mr.", "mrs.": "mr."}
_GENDER_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in _GENDER) + r")(?=\W|$)", re.I)


def _keep_case(src: str, dst: str) -> str:
    return dst.capitalize() if src[:1].isupper() else dst


def gender_swap(t: str) -> str:
    return _GENDER_RE.sub(lambda m: _keep_case(m.group(0), _GENDER[m.group(0).lower()]), t)


_US_JUR = re.compile(r"\b(the )?(state of )?(delaware|new york|california|texas|nevada|illinois|"
                     r"massachusetts|new jersey|florida|pennsylvania|ohio|georgia|washington|"
                     r"maryland|colorado|minnesota|north carolina|virginia)\b", re.I)


def eu_jurisdiction(t: str) -> str:
    """US state -> European jurisdiction (rotating so we test several)."""
    targets = ["England and Wales", "Germany", "France", "the Netherlands", "Ireland"]
    i = sum(map(ord, t)) % len(targets)
    t = _US_JUR.sub(targets[i], t)
    return re.sub(r"\b(United States|U\.S\.)\b", "European Union", t)


def euro(t: str) -> str:
    t = re.sub(r"\$\s?(\d)", r"EUR \1", t)
    return re.sub(r"\bdollars\b", "euros", re.sub(r"\bDollars\b", "Euros", t))


_BRIT = [("organization", "organisation"), ("authoriz", "authoris"), ("recogniz", "recognis"),
         ("favor", "favour"), ("labor", "labour"), ("license", "licence"), ("defense", "defence"),
         ("judgment", "judgement"), ("center", "centre"), ("canceled", "cancelled"),
         ("program", "programme"), ("fulfill", "fulfil"), ("analyz", "analys"), ("utiliz", "utilis")]


def british(t: str) -> str:
    for a, b in _BRIT:
        t = re.sub(a, b, t)
        t = re.sub(a.capitalize(), b.capitalize(), t)
    return t


def eu_party_names(t: str) -> str:
    """Generic US party designations -> fictitious European company names."""
    t = re.sub(r"\bthe Company\b", "Lumen Technik GmbH", t)
    t = re.sub(r"\bthe Borrower\b", "Rivière Énergie SAS", t)
    return re.sub(r"\bInc\.", "B.V.", t)


def typos(t: str, rate: float = 0.03, seed: int = SEED) -> str:
    """Drop ~3% of letters (OCR / scanning noise)."""
    rng = np.random.default_rng(seed + len(t))
    return "".join(c for c in t if not (c.isalpha() and rng.random() < rate))


PERTURBATIONS = {
    "gender swap (he<->she)": (gender_swap, lambda t: bool(_GENDER_RE.search(t))),
    "US state -> EU jurisdiction": (eu_jurisdiction, lambda t: bool(_US_JUR.search(t))),
    "$ -> EUR": (euro, lambda t: bool(re.search(r"\$\s?\d|\bdollars\b", t, re.I))),
    "British spelling": (british, lambda t: british(t) != t),
    "European party names": (eu_party_names, lambda t: eu_party_names(t) != t),
    "3% letter drops (OCR noise)": (typos, lambda t: True),
}

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


class Classifier:
    def __init__(self):
        self.device = get_device()
        self.tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(self.device)
        self.T = load_json(REPORTS / "eval_metrics.json")["calibration"]["temperature"]

    def probs(self, texts: list[str]) -> np.ndarray:
        max_len = load_json(MODEL_DIR / "training_config.json")["max_length"]
        lg = predict_logits(self.model, tokenize(self.tok, texts, max_len), self.tok.pad_token_id,
                            self.device, desc="audit")
        return softmax(lg, self.T)


def _wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def slice_report(y, pred, bpred, slices: dict[str, np.ndarray], per_class_recall: np.ndarray) -> dict:
    overall = float((pred == y).mean())
    out = {"overall_accuracy": overall}
    for name, arr in slices.items():
        rows = []
        for g in sorted(set(arr.tolist())):
            m = arr == g
            n, k = int(m.sum()), int((pred[m] == y[m]).sum())
            # What accuracy would we expect from this slice's clause-type mix alone?
            expected = float(per_class_recall[y[m]].mean())
            acc = k / n
            ci = _wilson(k, n)
            rows.append({
                "group": g, "n": n, "accuracy": acc, "accuracy_95ci": ci,
                "baseline_accuracy": float((bpred[m] == y[m]).mean()),
                "expected_from_label_mix": expected,
                "mix_adjusted_gap": acc - expected,
                "flag": bool(ci[1] < expected),  # significantly worse than its mix predicts
            })
        out[name] = rows
    return out


def run() -> dict:
    names = label_names()
    Xtr, ytr = split_arrays("train")
    Xte, yte = split_arrays("test")
    lt, _ = model_logits("test")
    pred = lt.argmax(1)
    bpred = np.load(ARTIFACTS / "baseline_test_logits.npz")["logits"].argmax(1)
    eval_m = load_json(REPORTS / "eval_metrics.json")
    thr = eval_m["calibration"]["abstain_threshold_for_95pct_val_accuracy"]

    recall = np.array([((pred == c) & (yte == c)).sum() / max((yte == c).sum(), 1) for c in range(len(names))])
    slices = compute_slices(Xte)
    buckets = frequency_buckets(ytr, len(names))
    slices["class_frequency"] = np.array([buckets[c] for c in yte])

    out: dict = {"slices": slice_report(yte, pred, bpred, slices, recall)}

    # ---- counterfactuals -------------------------------------------------
    clf = Classifier()
    cf = []
    examples = {}
    for pname, (fn, applies) in PERTURBATIONS.items():
        idx = [i for i, t in enumerate(Xte) if applies(t)]
        if pname.startswith("3%"):
            idx = np.random.default_rng(SEED).choice(len(Xte), 1000, replace=False).tolist()
        orig = [Xte[i] for i in idx]
        new = [fn(t) for t in orig]
        changed = [j for j, (a, b) in enumerate(zip(orig, new)) if a != b]
        if not changed:
            continue
        p0 = clf.probs([orig[j] for j in changed])
        p1 = clf.probs([new[j] for j in changed])
        y = yte[[idx[j] for j in changed]]
        flips = p0.argmax(1) != p1.argmax(1)
        cf.append({
            "perturbation": pname, "n": len(changed),
            "flip_rate": float(flips.mean()),
            "flip_rate_95ci": _wilson(int(flips.sum()), len(changed)),
            "accuracy_before": float((p0.argmax(1) == y).mean()),
            "accuracy_after": float((p1.argmax(1) == y).mean()),
            "mean_abs_conf_change_on_true_label": float(np.abs(p0[np.arange(len(y)), y] - p1[np.arange(len(y)), y]).mean()),
        })
        flipped = np.where(flips)[0][:3]
        examples[pname] = [{"before": orig[changed[j]][:250], "after": new[changed[j]][:250],
                            "pred_before": names[p0[j].argmax()], "pred_after": names[p1[j].argmax()],
                            "true": names[y[j]]} for j in flipped]
    out["counterfactuals"] = cf
    out["counterfactual_flip_examples"] = examples

    # ---- out-of-scope ----------------------------------------------------
    p = clf.probs([t for _, t in OUT_OF_SCOPE])
    out["out_of_scope"] = {
        "abstain_threshold": thr,
        "items": [{"input": n, "predicted": names[r.argmax()], "confidence": float(r.max()),
                   "abstains": bool(r.max() < thr)} for (n, _), r in zip(OUT_OF_SCOPE, p)],
    }

    out["retrieval"] = retrieval_audit(Xte, yte, slices)
    save_json(out, REPORTS / "audit.json")
    _figures(out)
    return out


def retrieval_audit(Xte, yte, slices) -> dict:
    from .retrieval import ClauseIndex, norm_hash

    idx = ClauseIndex.load()
    rng = np.random.default_rng(SEED)
    fam = slices["contract_family"]
    bank_fam = compute_slices(idx.texts)["contract_family"]
    res: dict = {}

    # Quality (hybrid P@10, clause->clause) per contract family, 150 queries each.
    sel = []
    for g in sorted(set(fam.tolist())):
        pool = np.where(fam == g)[0]
        sel += rng.choice(pool, min(150, len(pool)), replace=False).tolist()
    qemb = idx.embed([Xte[i] for i in sel], is_question=False)
    per_fam: dict[str, list[float]] = {}
    exposure: dict[str, int] = {}
    for i, e in zip(sel, qemb):
        s = idx.rrf(idx.dense_scores(e), idx.bm25_scores(Xte[i]))
        h = norm_hash(Xte[i])
        top = [j for j in np.argsort(-s)[:11] if idx.hashes[j] != h][:10]
        per_fam.setdefault(fam[i], []).append(float((idx.labels[top] == yte[i]).mean()))
        for j in top:
            if bank_fam[j] != fam[i]:
                exposure[bank_fam[j]] = exposure.get(bank_fam[j], 0) + 1
    res["hybrid_p@10_by_contract_family"] = {k: float(np.mean(v)) for k, v in per_fam.items()}
    bank_share = {g: float((bank_fam == g).mean()) for g in sorted(set(bank_fam.tolist()))}
    tot = sum(exposure.values()) or 1
    res["cross_family_results_share_vs_bank_share"] = {
        g: {"share_of_cross_family_results": exposure.get(g, 0) / tot, "share_of_bank": bank_share[g]}
        for g in bank_share}

    # Stability of top-10 under gender swap.
    gi = [i for i, t in enumerate(Xte) if _GENDER_RE.search(t)][:300]
    a = idx.embed([Xte[i] for i in gi], is_question=False)
    b = idx.embed([gender_swap(Xte[i]) for i in gi], is_question=False)
    jac = []
    for i, ea, eb in zip(gi, a, b):
        ta = set(np.argsort(-idx.rrf(idx.dense_scores(ea), idx.bm25_scores(Xte[i])))[:10])
        tb = set(np.argsort(-idx.rrf(idx.dense_scores(eb), idx.bm25_scores(gender_swap(Xte[i]))))[:10])
        jac.append(len(ta & tb) / len(ta | tb))
    res["gender_swap_top10_jaccard"] = {"mean": float(np.mean(jac)), "n": len(jac),
                                        "share_identical": float(np.mean(np.array(jac) == 1))}
    # Pronoun make-up of the bank (what a lawyer sees as "precedent").
    pr = compute_slices(idx.texts)["pronouns"]
    res["bank_pronoun_share"] = {g: float((pr == g).mean()) for g in sorted(set(pr.tolist()))}
    return res


def _figures(out):
    plt = plotting.plt
    panels = [k for k in ("contract_family", "jurisdiction", "pronouns", "length", "class_frequency")]
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 3.6), sharey=True)
    for ax, k in zip(axes, panels):
        rows = out["slices"][k]
        x = np.arange(len(rows))
        acc = [r["accuracy"] for r in rows]
        lo = [r["accuracy"] - r["accuracy_95ci"][0] for r in rows]
        hi = [r["accuracy_95ci"][1] - r["accuracy"] for r in rows]
        ax.bar(x, acc, color=plotting.SERIES[0], width=0.6, label="Accuracy (95% CI)")
        ax.errorbar(x, acc, yerr=[lo, hi], fmt="none", ecolor=plotting.INK_2, lw=1, capsize=2)
        ax.scatter(x, [r["expected_from_label_mix"] for r in rows], marker="_", s=300, lw=2,
                   color=plotting.SERIES[1], zorder=3, label="Expected from clause-type mix")
        ax.set_xticks(x, [f"{r['group']}\n(n={r['n']})" for r in rows], fontsize=7, rotation=30, ha="right")
        ax.set_title(k.replace("_", " "), fontsize=10)
        ax.set_ylim(0.5, 1.0)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Test accuracy")
    axes[0].legend(loc="lower left", fontsize=7)
    fig.suptitle("Classifier accuracy by slice", x=0.01, ha="left", fontweight="semibold")
    plotting.save(fig, FIGURES / "slice_accuracy.png")

    cf = out["counterfactuals"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    y = np.arange(len(cf))
    fr = [r["flip_rate"] for r in cf]
    ax.barh(y, fr, color=plotting.SERIES[0], height=0.55)
    ax.errorbar(fr, y, xerr=[[r["flip_rate"] - r["flip_rate_95ci"][0] for r in cf],
                             [r["flip_rate_95ci"][1] - r["flip_rate"] for r in cf]],
                fmt="none", ecolor=plotting.INK_2, lw=1, capsize=2)
    for yi, r in zip(y, cf):
        ax.text(r["flip_rate_95ci"][1] + 0.003, yi, f"{r['flip_rate']:.1%}  (n={r['n']})", va="center",
                fontsize=8, color=plotting.INK)
    ax.set_yticks(y, [r["perturbation"] for r in cf])
    ax.invert_yaxis()
    ax.set_xlabel("Share of clauses whose predicted type changes")
    ax.set_title("Counterfactual edits that should not change the clause type")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(r["flip_rate_95ci"][1] for r in cf) * 1.35 + 0.01)
    plotting.save(fig, FIGURES / "counterfactual_flips.png")


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2)[:6000])
