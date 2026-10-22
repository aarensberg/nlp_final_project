# Démonstration (environ 3 min)

**Prérequis** : `uv sync --frozen`, les artefacts présents dans `artifacts/` (poids, `baseline.joblib`, `inference_config.json`, index), et un terminal ouvert à la racine du projet.

**Vérification la veille** (~1 min 30) :

```bash
uv run pytest tests/test_artifacts.py -q
```

Les sorties réelles des commandes ci-dessous sont enregistrées dans [demo_outputs/](demo_outputs/). Ce sont elles qui servent de **plan B**.

## 1. Classification correcte, sortie JSON (~12 s, surtout le chargement du modèle)

```bash
uv run nlp-final-project classify --json "This Agreement shall be governed by and construed in accordance with the laws of the State of New York, without regard to its conflict of laws principles."
```

Sortie attendue ([1_classify_governing_law.json](demo_outputs/1_classify_governing_law.json)) :
- `"label": "Governing Laws"`, `"confidence": 0.974`, `"abstain": false`, avec un seuil de 0,709 ;
- 2ᵉ candidat : *Applicable Laws* (0,023), un label quasi synonyme ;
- une note rappelle que le score de confiance n'est **pas** une probabilité de validité juridique.

**À dire** : la sortie JSON est sur stdout ; les messages techniques vont sur stderr.

## 2. Abstention sur une clause hors périmètre (~10 s)

```bash
uv run nlp-final-project classify "Le présent contrat est régi par le droit français."
```

Sortie attendue ([2_classify_french_abstains.txt](demo_outputs/2_classify_french_abstains.txt)) : `ABSTAIN`, confiance 0,033, à envoyer à un juriste.

**À dire** : cette fois l'abstention fonctionne, mais un seuil ne garantit pas de détecter toutes les entrées hors périmètre. BERT seul classe une clause RGPD avec 0,90 de confiance (audit).

## 3. Recherche réussie (~16 s)

```bash
uv run nlp-final-project search "either party may terminate on thirty days written notice" -k 5
```

Sortie attendue ([3_search_success.txt](demo_outputs/3_search_success.txt)) : 5 clauses *Terminations*, chacune avec un préavis de « thirty (30) days ». Le filtre annonce 824 clauses admissibles.

## 4. Échec contrôlé : la question « 30-day notice » (~16 s)

```bash
uv run nlp-final-project search "show me termination clauses with a 30-day notice period" -k 5
```

Sortie attendue ([4_search_30day_controlled_failure.txt](demo_outputs/4_search_30day_controlled_failure.txt)) :
- 1er résultat *Notices* : 30 jours pour notifier une « Good Reason », **pas** un préavis de résiliation ;
- résultats 2 et 3 : *Notices* et *Terms* ; le premier *Terminations* n'arrive qu'en 4ᵉ position ;
- le message du filtre le dit lui-même : « Mentioning a duration does not mean it is the requested notice period ».

**À dire** : la recherche dépend de la formulation, et la démo 3 ne la valide pas. La correction passerait par une extraction structurée (rôle + durée).

## 5. D'où viennent les chiffres (~5 s)

```bash
uv run python -c "import json; d=json.load(open('reports/eval_metrics.json')); [print(v, r['test']['micro_f1'], r['test']['macro_f1'], r['abstention']['test_coverage']) for v, r in d['variants'].items()]"
```

Sortie attendue : `ensemble 0.8843 0.828…  0.842`, `bert 0.8635 0.762… 0.7637`, `tfidf 0.877 0.821… 0.8321`.

## Plan B, si le modèle ne charge pas

1. Ouvrir les fichiers de [demo_outputs/](demo_outputs/), qui sont les sorties réelles enregistrées.
2. Montrer [reports/retrieval_questions.json](../reports/retrieval_questions.json) : chercher `show me termination`, mode `hybrid + period filter`.
3. Montrer les figures de `reports/figures/` et [key_numbers.md](key_numbers.md).
4. Dire explicitement qu'il s'agit de sorties enregistrées, pas d'une exécution en direct.
