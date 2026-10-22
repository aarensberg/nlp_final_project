# Chiffres clés (généré)

Généré par `uv run python scripts/key_numbers.py` à partir des rapports versionnés : aucune valeur saisie à la main.
Validation = choix des paramètres ; test = consulté pour évaluer, jamais pour régler ; les 40 questions de recherche ont déjà été vues (non-régression) ; l'audit utilise des proxys textuels.
Macro-F1 principal : univers fixe de 100 labels, `zero_division=0`. Les macro-F1 de `train_log.json` utilisent l'ancienne définition de l'époque d'entraînement (labels présents seulement) : ce sont des traces historiques.


## Données

| Grandeur | Valeur | Source |
|---|---|---|
| Clauses train / validation / test | {'train': 60000, 'test': 10000, 'validation': 10000} | `reports/eda.json` → `splits` |
| Classe la plus fréquente (train) | ['Governing Laws', 3167] | `reports/eda.json` → `imbalance/most_frequent/0` |
| Classe la plus rare (train) | ['Books', 23] | `reports/eda.json` → `imbalance/least_frequent/9` |
| Clauses du test identiques à une clause du train | 282 | `reports/eda.json` → `duplicates/test_texts_also_in_train` |
| Part des clauses > 256 tokens (échantillon 5k) | 11.90 % | `reports/eda.json` → `lengths/share_over_256_tokens` |

## Entraînement (trace, ancienne déf. macro)

| Grandeur | Valeur | Source |
|---|---|---|
| Validation époque 1 micro-F1 | 84.18 % | `reports/train_log.json` → `epochs/0/micro_f1` |
| Validation époque 1 macro-F1 (labels présents) | 74.02 % | `reports/train_log.json` → `epochs/0/macro_f1` |
| Validation époque 2 micro-F1 | 86.37 % | `reports/train_log.json` → `epochs/1/micro_f1` |
| Validation époque 2 macro-F1 (labels présents) | 77.58 % | `reports/train_log.json` → `epochs/1/macro_f1` |
| Durée (min) | 146.497 | `reports/train_log.json` → `train_minutes` |

## Paramètres (validation)

| Grandeur | Valeur | Source |
|---|---|---|
| Poids BERT dans la combinaison | 0.2 | `reports/inference_config.json` → `ensemble/weight_bert` |
| Température BERT | 1.0369 | `reports/inference_config.json` → `bert/temperature` |
| Température TF-IDF | 0.9053 | `reports/inference_config.json` → `tfidf/temperature` |
| Seuil d'abstention ensemble | 0.7089 | `reports/inference_config.json` → `abstention/thresholds/ensemble` |
| Couverture validation ensemble | 84.06 % | `reports/inference_config.json` → `abstention/validation_coverage/ensemble` |
| Seuil d'abstention bert | 0.8508 | `reports/inference_config.json` → `abstention/thresholds/bert` |
| Couverture validation bert | 76.47 % | `reports/inference_config.json` → `abstention/validation_coverage/bert` |
| Seuil d'abstention tfidf | 0.7562 | `reports/inference_config.json` → `abstention/thresholds/tfidf` |
| Couverture validation tfidf | 83.26 % | `reports/inference_config.json` → `abstention/validation_coverage/tfidf` |

## Validation (macro standard)

| Grandeur | Valeur | Source |
|---|---|---|
| ensemble micro-F1 | 88.78 % | `reports/eval_metrics.json` → `variants/ensemble/validation/micro_f1` |
| ensemble macro-F1 | 82.90 % | `reports/eval_metrics.json` → `variants/ensemble/validation/macro_f1` |
| bert micro-F1 | 86.37 % | `reports/eval_metrics.json` → `variants/bert/validation/micro_f1` |
| bert macro-F1 | 76.80 % | `reports/eval_metrics.json` → `variants/bert/validation/macro_f1` |
| tfidf micro-F1 | 87.97 % | `reports/eval_metrics.json` → `variants/tfidf/validation/micro_f1` |
| tfidf macro-F1 | 81.88 % | `reports/eval_metrics.json` → `variants/tfidf/validation/macro_f1` |

## Test

| Grandeur | Valeur | Source |
|---|---|---|
| ensemble micro-F1 | 88.43 % | `reports/eval_metrics.json` → `variants/ensemble/test/micro_f1` |
| ensemble IC 95 % micro-F1 | [87.78 ; 89.06] | `reports/eval_metrics.json` → `variants/ensemble/test/micro_f1_95ci` |
| ensemble macro-F1 | 82.84 % | `reports/eval_metrics.json` → `variants/ensemble/test/macro_f1` |
| ensemble IC 95 % macro-F1 | [81.17 ; 83.68] | `reports/eval_metrics.json` → `variants/ensemble/test/macro_f1_95ci` |
| ensemble macro-F1 sans doublons train | 82.49 % | `reports/eval_metrics.json` → `variants/ensemble/test_without_train_duplicates/macro_f1` |
| ensemble macro-F1 classes de queue | 74.78 % | `reports/eval_metrics.json` → `variants/ensemble/macro_f1_by_frequency_bucket/tail` |
| ensemble ECE (calibrée) | 0.0167 | `reports/eval_metrics.json` → `variants/ensemble/calibration/ece_test` |
| ensemble couverture avec abstention | 84.20 % | `reports/eval_metrics.json` → `variants/ensemble/abstention/test_coverage` |
| ensemble correctes parmi les clauses traitées | 94.51 % | `reports/eval_metrics.json` → `variants/ensemble/abstention/test_correct_rate_among_handled` |
| bert micro-F1 | 86.35 % | `reports/eval_metrics.json` → `variants/bert/test/micro_f1` |
| bert IC 95 % micro-F1 | [85.66 ; 87.02] | `reports/eval_metrics.json` → `variants/bert/test/micro_f1_95ci` |
| bert macro-F1 | 76.27 % | `reports/eval_metrics.json` → `variants/bert/test/macro_f1` |
| bert IC 95 % macro-F1 | [75.18 ; 77.03] | `reports/eval_metrics.json` → `variants/bert/test/macro_f1_95ci` |
| bert macro-F1 sans doublons train | 76.10 % | `reports/eval_metrics.json` → `variants/bert/test_without_train_duplicates/macro_f1` |
| bert macro-F1 classes de queue | 61.89 % | `reports/eval_metrics.json` → `variants/bert/macro_f1_by_frequency_bucket/tail` |
| bert ECE (calibrée) | 0.0198 | `reports/eval_metrics.json` → `variants/bert/calibration/ece_test` |
| bert couverture avec abstention | 76.37 % | `reports/eval_metrics.json` → `variants/bert/abstention/test_coverage` |
| bert correctes parmi les clauses traitées | 94.83 % | `reports/eval_metrics.json` → `variants/bert/abstention/test_correct_rate_among_handled` |
| tfidf micro-F1 | 87.70 % | `reports/eval_metrics.json` → `variants/tfidf/test/micro_f1` |
| tfidf IC 95 % micro-F1 | [87.02 ; 88.37] | `reports/eval_metrics.json` → `variants/tfidf/test/micro_f1_95ci` |
| tfidf macro-F1 | 82.19 % | `reports/eval_metrics.json` → `variants/tfidf/test/macro_f1` |
| tfidf IC 95 % macro-F1 | [80.52 ; 83.00] | `reports/eval_metrics.json` → `variants/tfidf/test/macro_f1_95ci` |
| tfidf macro-F1 sans doublons train | 81.83 % | `reports/eval_metrics.json` → `variants/tfidf/test_without_train_duplicates/macro_f1` |
| tfidf macro-F1 classes de queue | 74.28 % | `reports/eval_metrics.json` → `variants/tfidf/macro_f1_by_frequency_bucket/tail` |
| tfidf ECE (calibrée) | 0.0242 | `reports/eval_metrics.json` → `variants/tfidf/calibration/ece_test` |
| tfidf couverture avec abstention | 83.21 % | `reports/eval_metrics.json` → `variants/tfidf/abstention/test_coverage` |
| tfidf correctes parmi les clauses traitées | 94.59 % | `reports/eval_metrics.json` → `variants/tfidf/abstention/test_correct_rate_among_handled` |

## Test (sensibilité, jamais principal)

| Grandeur | Valeur | Source |
|---|---|---|
| ensemble macro-F1 si classe absente = 1 | 82.84 % | `reports/eval_metrics.json` → `variants/ensemble/sensitivity_macro_f1_absent_both_counts_one/test/macro_f1_absent_one` |
| bert macro-F1 si classe absente = 1 | 76.27 % | `reports/eval_metrics.json` → `variants/bert/sensitivity_macro_f1_absent_both_counts_one/test/macro_f1_absent_one` |
| tfidf macro-F1 si classe absente = 1 | 82.19 % | `reports/eval_metrics.json` → `variants/tfidf/sensitivity_macro_f1_absent_both_counts_one/test/macro_f1_absent_one` |

## Test (comparaisons appariées)

| Grandeur | Valeur | Source |
|---|---|---|
| ensemble_vs_tfidf Δ micro-F1 | +0.73 pt | `reports/eval_metrics.json` → `comparisons/ensemble_vs_tfidf/delta_micro_f1_points` |
| ensemble_vs_tfidf IC Δ micro-F1 | [+0.51 ; +0.98] pt | `reports/eval_metrics.json` → `comparisons/ensemble_vs_tfidf/delta_micro_f1_95ci_points` |
| ensemble_vs_tfidf Δ macro-F1 | +0.65 pt | `reports/eval_metrics.json` → `comparisons/ensemble_vs_tfidf/delta_macro_f1_points` |
| ensemble_vs_tfidf IC Δ macro-F1 | [+0.31 ; +1.00] pt | `reports/eval_metrics.json` → `comparisons/ensemble_vs_tfidf/delta_macro_f1_95ci_points` |
| ensemble_vs_tfidf McNemar p | 3.979392264405063e-10 | `reports/eval_metrics.json` → `comparisons/ensemble_vs_tfidf/mcnemar/p_value` |
| ensemble_vs_bert Δ micro-F1 | +2.08 pt | `reports/eval_metrics.json` → `comparisons/ensemble_vs_bert/delta_micro_f1_points` |
| ensemble_vs_bert IC Δ micro-F1 | [+1.54 ; +2.60] pt | `reports/eval_metrics.json` → `comparisons/ensemble_vs_bert/delta_micro_f1_95ci_points` |
| ensemble_vs_bert Δ macro-F1 | +6.57 pt | `reports/eval_metrics.json` → `comparisons/ensemble_vs_bert/delta_macro_f1_points` |
| ensemble_vs_bert IC Δ macro-F1 | [+5.20 ; +7.51] pt | `reports/eval_metrics.json` → `comparisons/ensemble_vs_bert/delta_macro_f1_95ci_points` |
| ensemble_vs_bert McNemar p | 9.468421373002124e-15 | `reports/eval_metrics.json` → `comparisons/ensemble_vs_bert/mcnemar/p_value` |
| bert_vs_tfidf Δ micro-F1 | -1.35 pt | `reports/eval_metrics.json` → `comparisons/bert_vs_tfidf/delta_micro_f1_points` |
| bert_vs_tfidf IC Δ micro-F1 | [-1.93 ; -0.77] pt | `reports/eval_metrics.json` → `comparisons/bert_vs_tfidf/delta_micro_f1_95ci_points` |
| bert_vs_tfidf Δ macro-F1 | -5.92 pt | `reports/eval_metrics.json` → `comparisons/bert_vs_tfidf/delta_macro_f1_points` |
| bert_vs_tfidf IC Δ macro-F1 | [-6.88 ; -4.50] pt | `reports/eval_metrics.json` → `comparisons/bert_vs_tfidf/delta_macro_f1_95ci_points` |
| bert_vs_tfidf McNemar p | 4.62918237575014e-06 | `reports/eval_metrics.json` → `comparisons/bert_vs_tfidf/mcnemar/p_value` |

## Recherche clause → clause (981 requêtes test)

| Grandeur | Valeur | Source |
|---|---|---|
| hybride p@1 | 77.37 % | `reports/retrieval_metrics.json` → `clause_to_clause/hybrid/p@1` |
| hybride p@5 | 70.11 % | `reports/retrieval_metrics.json` → `clause_to_clause/hybrid/p@5` |
| hybride p@10 | 66.35 % | `reports/retrieval_metrics.json` → `clause_to_clause/hybrid/p@10` |
| hybride mrr@10 | 0.8236 | `reports/retrieval_metrics.json` → `clause_to_clause/hybrid/mrr@10` |

## 40 questions DÉJÀ VUES (non-régression)

| Grandeur | Valeur | Source |
|---|---|---|
| dense P@1 | 85.00 % | `reports/retrieval_metrics.json` → `questions/0/p@1` |
| dense precision_at_10 | 87.50 % | `reports/retrieval_metrics.json` → `questions/0/precision_at_10` |
| bm25 P@1 | 82.50 % | `reports/retrieval_metrics.json` → `questions/1/p@1` |
| bm25 precision_at_10 | 83.00 % | `reports/retrieval_metrics.json` → `questions/1/precision_at_10` |
| hybrid P@1 | 92.50 % | `reports/retrieval_metrics.json` → `questions/2/p@1` |
| hybrid precision_at_10 | 89.25 % | `reports/retrieval_metrics.json` → `questions/2/precision_at_10` |
| hybrid + period filter P@1 | 95.00 % | `reports/retrieval_metrics.json` → `questions/3/p@1` |
| hybrid + period filter precision_at_10 | 89.00 % | `reports/retrieval_metrics.json` → `questions/3/precision_at_10` |
| Question vitrine « 30-day notice » : P@1 / precision_at_10 | 0 / 0.1 | `reports/retrieval_questions.json` → mode `hybrid + period filter` |
| Question vitrine : label du 1er résultat | Notices | `reports/retrieval_questions.json` → `hits/0/label` |

## Audit (proxys)

| Grandeur | Valeur | Source |
|---|---|---|
| Bail (n=266) : exactitude / attendu / correctes après abstention (combinaison) | 75.2 % / 84.3 % / 87.6 % | `reports/audit.json` → `slices/contract_family[real_estate]/ensemble` |
| Classes de queue : correctes après abstention (combinaison) | 84.3 % | `reports/audit.json` → `slices/class_frequency[tail]/ensemble` |

## Audit (contrefactuels, combinaison)

| Grandeur | Valeur | Source |
|---|---|---|
| gender swap (he<->she) | 2.3 % (n=487) | `reports/audit.json` → `counterfactuals[gender swap (he<->she)]/ensemble/type_flip_rate` |
| US state -> European jurisdiction | 3.6 % (n=698) | `reports/audit.json` → `counterfactuals[US state -> European jurisdiction]/ensemble/type_flip_rate` |
| $ -> EUR | 0.0 % (n=365) | `reports/audit.json` → `counterfactuals[$ -> EUR]/ensemble/type_flip_rate` |
| British spelling | 2.2 % (n=1275) | `reports/audit.json` → `counterfactuals[British spelling]/ensemble/type_flip_rate` |
| European party names | 0.8 % (n=3199) | `reports/audit.json` → `counterfactuals[European party names]/ensemble/type_flip_rate` |
| 3% letter drops (OCR noise) | 5.4 % (n=992) | `reports/audit.json` → `counterfactuals[3% letter drops (OCR noise)]/ensemble/type_flip_rate` |

## Audit (hors périmètre, 8 entrées)

| Grandeur | Valeur | Source |
|---|---|---|
| Erreurs confiantes combinaison / BERT / TF-IDF | 0 / 1 (GDPR data processing) / 0 | `reports/audit.json` → `out_of_scope/confident_errors` |

## Audit (recherche)

| Grandeur | Valeur | Source |
|---|---|---|
| P@10 hybride bail / crédit | 59.9 % / 79.6 % | `reports/audit.json` → `retrieval/hybrid_p@10_by_contract_family` |
