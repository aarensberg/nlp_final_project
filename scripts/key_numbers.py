"""Generate presentation/key_numbers.md from the versioned reports (no number typed by hand).

    uv run python scripts/key_numbers.py
"""

import json
from pathlib import Path

from nlp_final_project.config import REPORTS, ROOT

R = {name: json.loads((REPORTS / f"{name}.json").read_text())
     for name in ("eda", "train_log", "eval_metrics", "inference_config", "retrieval_metrics", "audit")}
Q = json.loads((REPORTS / "retrieval_questions.json").read_text())["results"]
rows: list[tuple[str, str, str, str]] = []  # section, label, value, source


def get(report: str, path: str):
    node = R[report]
    for part in path.split("/"):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def add(section, label, report, path, fmt="pct"):
    v = get(report, path)
    if fmt == "pct":
        s = f"{v * 100:.2f} %"
    elif fmt == "ci":
        s = f"[{v[0] * 100:.2f} ; {v[1] * 100:.2f}]"
    elif fmt == "pts":
        s = f"{v:+.2f} pt"
    elif fmt == "ci_pts":
        s = f"[{v[0]:+.2f} ; {v[1]:+.2f}] pt"
    elif fmt == "f3":
        s = f"{v:.3f}"
    elif fmt == "f4":
        s = f"{v:.4f}"
    else:
        s = str(v)
    rows.append((section, label, s, f"`reports/{report}.json` → `{path}`"))


# Data
add("Données", "Clauses train / validation / test", "eda", "splits", "raw")
add("Données", "Classe la plus fréquente (train)", "eda", "imbalance/most_frequent/0", "raw")
add("Données", "Classe la plus rare (train)", "eda", "imbalance/least_frequent/9", "raw")
add("Données", "Clauses du test identiques à une clause du train", "eda", "duplicates/test_texts_also_in_train", "raw")
add("Données", "Part des clauses > 256 tokens (échantillon 5k)", "eda", "lengths/share_over_256_tokens")
# Training (historical trace, training-time macro-F1 definition)
add("Entraînement (trace, ancienne déf. macro)", "Validation époque 1 micro-F1", "train_log", "epochs/0/micro_f1")
add("Entraînement (trace, ancienne déf. macro)", "Validation époque 1 macro-F1 (labels présents)", "train_log", "epochs/0/macro_f1")
add("Entraînement (trace, ancienne déf. macro)", "Validation époque 2 micro-F1", "train_log", "epochs/1/micro_f1")
add("Entraînement (trace, ancienne déf. macro)", "Validation époque 2 macro-F1 (labels présents)", "train_log", "epochs/1/macro_f1")
add("Entraînement (trace, ancienne déf. macro)", "Durée (min)", "train_log", "train_minutes", "f3")
# Validation-fitted parameters
add("Paramètres (validation)", "Poids BERT dans la combinaison", "inference_config", "ensemble/weight_bert", "raw")
add("Paramètres (validation)", "Température BERT", "inference_config", "bert/temperature", "f4")
add("Paramètres (validation)", "Température TF-IDF", "inference_config", "tfidf/temperature", "f4")
for v in ("ensemble", "bert", "tfidf"):
    add("Paramètres (validation)", f"Seuil d'abstention {v}", "inference_config", f"abstention/thresholds/{v}", "f4")
    add("Paramètres (validation)", f"Couverture validation {v}", "inference_config", f"abstention/validation_coverage/{v}")
# Validation scores
for v in ("ensemble", "bert", "tfidf"):
    add("Validation (macro standard)", f"{v} micro-F1", "eval_metrics", f"variants/{v}/validation/micro_f1")
    add("Validation (macro standard)", f"{v} macro-F1", "eval_metrics", f"variants/{v}/validation/macro_f1")
# Test
for v in ("ensemble", "bert", "tfidf"):
    b = f"variants/{v}"
    add("Test", f"{v} micro-F1", "eval_metrics", f"{b}/test/micro_f1")
    add("Test", f"{v} IC 95 % micro-F1", "eval_metrics", f"{b}/test/micro_f1_95ci", "ci")
    add("Test", f"{v} macro-F1", "eval_metrics", f"{b}/test/macro_f1")
    add("Test", f"{v} IC 95 % macro-F1", "eval_metrics", f"{b}/test/macro_f1_95ci", "ci")
    add("Test", f"{v} macro-F1 sans doublons train", "eval_metrics", f"{b}/test_without_train_duplicates/macro_f1")
    add("Test", f"{v} macro-F1 classes de queue", "eval_metrics", f"{b}/macro_f1_by_frequency_bucket/tail")
    add("Test", f"{v} ECE (calibrée)", "eval_metrics", f"{b}/calibration/ece_test", "f4")
    add("Test", f"{v} couverture avec abstention", "eval_metrics", f"{b}/abstention/test_coverage")
    add("Test", f"{v} correctes parmi les clauses traitées", "eval_metrics",
        f"{b}/abstention/test_correct_rate_among_handled")
    add("Test (sensibilité, jamais principal)", f"{v} macro-F1 si classe absente = 1", "eval_metrics",
        f"{b}/sensitivity_macro_f1_absent_both_counts_one/test/macro_f1_absent_one")
for c in ("ensemble_vs_tfidf", "ensemble_vs_bert", "bert_vs_tfidf"):
    add("Test (comparaisons appariées)", f"{c} Δ micro-F1", "eval_metrics", f"comparisons/{c}/delta_micro_f1_points", "pts")
    add("Test (comparaisons appariées)", f"{c} IC Δ micro-F1", "eval_metrics", f"comparisons/{c}/delta_micro_f1_95ci_points", "ci_pts")
    add("Test (comparaisons appariées)", f"{c} Δ macro-F1", "eval_metrics", f"comparisons/{c}/delta_macro_f1_points", "pts")
    add("Test (comparaisons appariées)", f"{c} IC Δ macro-F1", "eval_metrics", f"comparisons/{c}/delta_macro_f1_95ci_points", "ci_pts")
    add("Test (comparaisons appariées)", f"{c} McNemar p", "eval_metrics", f"comparisons/{c}/mcnemar/p_value", "raw")
# Retrieval
for k in ("p@1", "p@5", "p@10", "mrr@10"):
    add("Recherche clause → clause (981 requêtes test)", f"hybride {k}", "retrieval_metrics",
        f"clause_to_clause/hybrid/{k}", "f4" if k == "mrr@10" else "pct")
for i, mode in enumerate(("dense", "bm25", "hybrid", "hybrid + period filter")):
    add("40 questions DÉJÀ VUES (non-régression)", f"{mode} P@1", "retrieval_metrics", f"questions/{i}/p@1")
    add("40 questions DÉJÀ VUES (non-régression)", f"{mode} precision_at_10", "retrieval_metrics", f"questions/{i}/precision_at_10")
head = [x for x in Q if x["mode"] == "hybrid + period filter" and x["question"].startswith("show me termination")][0]
rows.append(("40 questions DÉJÀ VUES (non-régression)", "Question vitrine « 30-day notice » : P@1 / precision_at_10",
             f"{head['p@1']:.0f} / {head['precision_at_10']:.1f}", "`reports/retrieval_questions.json` → mode `hybrid + period filter`"))
rows.append(("40 questions DÉJÀ VUES (non-régression)", "Question vitrine : label du 1er résultat",
             head["hits"][0]["label"], "`reports/retrieval_questions.json` → `hits/0/label`"))
# Audit
slices = {r["group"]: r for r in R["audit"]["slices"]["contract_family"]}
re_ = slices["real_estate"]["ensemble"]
rows.append(("Audit (proxys)", "Bail (n=266) : exactitude / attendu / correctes après abstention (combinaison)",
             f"{re_['accuracy']*100:.1f} % / {re_['expected_from_clause_type_mix']*100:.1f} % / {re_['correct_rate_among_handled']*100:.1f} %",
             "`reports/audit.json` → `slices/contract_family[real_estate]/ensemble`"))
tail = {r["group"]: r for r in R["audit"]["slices"]["class_frequency"]}["tail"]["ensemble"]
rows.append(("Audit (proxys)", "Classes de queue : correctes après abstention (combinaison)",
             f"{tail['correct_rate_among_handled']*100:.1f} %", "`reports/audit.json` → `slices/class_frequency[tail]/ensemble`"))
for r in R["audit"]["counterfactuals"]:
    rows.append(("Audit (contrefactuels, combinaison)", r["perturbation"],
                 f"{r['ensemble']['type_flip_rate']*100:.1f} % (n={r['n']})",
                 f"`reports/audit.json` → `counterfactuals[{r['perturbation']}]/ensemble/type_flip_rate`"))
ce = R["audit"]["out_of_scope"]["confident_errors"]
rows.append(("Audit (hors périmètre, 8 entrées)", "Erreurs confiantes combinaison / BERT / TF-IDF",
             f"{len(ce['ensemble'])} / {len(ce['bert'])} ({', '.join(ce['bert']) or '-'}) / {len(ce['tfidf'])}",
             "`reports/audit.json` → `out_of_scope/confident_errors`"))
fam = R["audit"]["retrieval"]["hybrid_p@10_by_contract_family"]
rows.append(("Audit (recherche)", "P@10 hybride bail / crédit",
             f"{fam['real_estate']['p@10']*100:.1f} % / {fam['credit_finance']['p@10']*100:.1f} %",
             "`reports/audit.json` → `retrieval/hybrid_p@10_by_contract_family`"))

out = ["# Chiffres clés (généré)", "",
       "Généré par `uv run python scripts/key_numbers.py` à partir des rapports versionnés : aucune valeur saisie à la main.",
       "Validation = choix des paramètres ; test = consulté pour évaluer, jamais pour régler ; les 40 questions de recherche ont déjà été vues (non-régression) ; l'audit utilise des proxys textuels.",
       "Macro-F1 principal : univers fixe de 100 labels, `zero_division=0`. Les macro-F1 de `train_log.json` utilisent l'ancienne définition de l'époque d'entraînement (labels présents seulement) : ce sont des traces historiques.", ""]
grouped: dict[str, list] = {}
for sec, label, val, src in rows:  # keep first-appearance order of sections
    grouped.setdefault(sec, []).append((label, val, src))
for sec, items in grouped.items():
    out += ["", f"## {sec}", "", "| Grandeur | Valeur | Source |", "|---|---|---|"]
    out += [f"| {label} | {val} | {src} |" for label, val, src in items]
dest = ROOT / "presentation" / "key_numbers.md"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text("\n".join(out) + "\n")
print(f"wrote {dest.relative_to(ROOT)} ({len(rows)} numbers)")
