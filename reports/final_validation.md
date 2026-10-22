# Validation finale avant remise

Données structurées : [final_validation.json](final_validation.json), généré par `uv run python scripts/final_validation.py`. Journaux bruts, avec chemins pseudonymisés : [validation_logs/](validation_logs/).

## Verdict

**Prêt pour la remise**, avec les limites documentées ci-dessous.
- **Tests :** les 92 tests passent, sans échec ni test ignoré, dont 12 sur les artefacts réels.
- **Cohérence :** les 15 vérifications croisées entre JSON, CSV, configuration et manifeste passent.
- **Reconstruction propre :** réussie sur la même machine.
- **Prédictions :** aucune n'a changé et Legal-BERT n'a pas été réentraîné. Seuls des intervalles de confiance et une définition ont changé, pour les raisons expliquées plus bas.

Non démontré, à dire à l'oral :
- la reproductibilité sur une autre machine ;
- l'identité d'une reprise d'entraînement sur MPS ;
- la variance entre graines.

## Git

| | Valeur |
|---|---|
| Commit de départ | `1c2b565` (fix(nlp): stabilize evaluation, retrieval and reproducibility) |
| État final | arbre **non commité** : 24 entrées dans `git status --short`. `final_validation.json` → `working_tree` en liste 23, car il a été généré juste avant ce fichier `.md`. |
| Poids, caches, embeddings | non versionnés (`artifacts/` ignoré) ; aucun fichier versionné de plus de 2 Mo |

## Défauts

| # | Défaut | Où (`1c2b565`) | Statut |
|---|---|---|---|
| 1 | Le macro-F1 comptait 1 pour un label absent de la vérité et des prédictions, ce qui gonflait les bootstraps et la validation (*Books* absent) | `metrics.py:26` | **Confirmé, corrigé** : `zero_division=0` sur `range(100)` partout ; ancienne convention gardée en sensibilité |
| 2 | Contradiction supposée sur la provenance (« jamais repris » contre `resume: true`) | `README.md:157`, `train_log.json` | **Non confirmée sur le fond** : `train.log` prouve un run unique de l'étape 0 à 7 500. `resume: true` enregistre seulement l'option passée. Formulation précisée et [training_provenance.md](training_provenance.md) créé |
| 3 | `--resume` sans checkpoint repartait silencieusement de zéro (cause du point 2) | `train.py:263` | **Confirmé, corrigé** : échec explicite ; nouvelle option `--resume-if-present` |
| 4 | Structure du modèle TF-IDF non vérifiée (seule l'empreinte était contrôlée) | `inference.py:117,222` | **Confirmé, corrigé** : contrôle de `classes_` (= 0..99) et du nombre de sorties (100) |
| 5 | L'index ne vérifiait que `embeddings.npy` | `retrieval.py:167,181` | **Confirmé, corrigé** : SHA-256, tailles et formes des 3 fichiers, longueurs égales ; reconstruction avec explication sur stderr |
| 6 | CSV écrit avec des fins de ligne `\r\n` (`git diff --check`) | `evaluate.py:163` | **Confirmé, corrigé** ; l'archive garde la copie exacte de `1c2b565` |
| 7 | Description du projet par défaut | `pyproject.toml:4` | **Corrigé** |
| 8 | Chemins personnels dans les journaux de validation et le manifeste, via le chemin du scratchpad | nouveaux fichiers | **Trouvé, corrigé** : pseudonymisation, et le manifeste rejette désormais `/Users/`, `/home/` et le nom d'utilisateur |
| 9 | Mes scripts tronquaient le 1er chemin de `git status` (`.strip()`) | `manifest.py`, `final_validation.py` | **Trouvé, corrigé** avant la clôture |
| 10 | Affirmation fausse dans le Q&A : « la combinaison est la seule à s'abstenir sur les 8 entrées » (TF-IDF aussi) | `presentation/jury_qa.md` | **Trouvé, corrigé** avant la remise |
| 11 | Dépendance circulaire : le manifeste empreintait `final_validation.json`, qui contient l'empreinte du manifeste (vérification croisée en échec) | `manifest.py` | **Trouvé, corrigé** : `final_validation.*` exclu du manifeste ; ordre de génération fixé (manifeste → validation → vérifications) |
| — | Chemin absolu personnel dans `reports/train_log.json` et les archives | traces historiques | **Non corrigé volontairement** : pas de réécriture rétroactive des traces. À décider (voir limites) |

## Commandes exécutées

Durées réelles, toutes avec code de retour 0. Le détail horodaté est dans `final_validation.json` → `commands`.

| Commande | Durée | Résultat |
|---|---|---|
| Tests initiaux sur `1c2b565` : sans artefacts / artefacts rapides / artefacts lents | 53 s / 10 s / 44 s | 58 / 8 / 3 réussis |
| Tests rapides après corrections | 60 s | 80 réussis |
| `nlp-final-project evaluate` (refit validation, logits réutilisés) | 16 s | poids et seuils inchangés |
| `nlp-final-project index` (reconstruction : ancien format de `meta.json`) | 298 s | 160 listes de résultats sur 160 identiques |
| `nlp-final-project audit` | 300 s | identique à l'archive |
| Tests avec artefacts : rapides / lents | 12 s / 43 s | 9 / 3 réussis |
| Reconstruction propre : `uv sync --frozen` + tests sans artefacts + tests avec artefacts liés | 1 s + 115 s + 94 s | 80 réussis + 12 ignorés ; puis 92 réussis ; artefacts inchangés (21 fichiers) |
| `evaluate` (CSV en fins de ligne Unix) | 14 s | `eval_metrics.json` identique octet par octet |
| Suite complète (clôture, après le dernier changement de code) | 95 s | **92 réussis** |
| `scripts/key_numbers.py` · `nlp-final-project manifest` · `scripts/cross_checks.py --manifest` | 1 s · 8 s · 9 s | 94 chiffres · manifeste · **15/15** |

## Tests par groupe (état final)

| Groupe | Artefacts | Réussis | Échoués | Ignorés |
|---|---|---|---|---|
| `test_units.py` + `test_integrity.py` + `test_cli.py` + `test_resume.py` | non | 80 | 0 | 0 |
| `test_artifacts.py` (non lents) | oui | 9 | 0 | 0 |
| `test_artifacts.py` (lents : BERT réel, CLI) | oui | 3 | 0 | 0 |
| **Total** | | **92** | **0** | **0** |

## Avant / après (`1c2b565` → actuel)

Seules 4 grandeurs changent. Toutes viennent de la **correction du macro-F1** : aucune prédiction, aucun poids, aucun seuil ne bouge.

| Grandeur | `1c2b565` | Actuel | Cause |
|---|---|---|---|
| IC 95 % macro-F1 test, combinaison | [81,57 ; 83,76] | **[81,17 ; 83,68]** | un label absent d'un rééchantillonnage compte désormais 0 |
| IC 95 % macro-F1 test, TF-IDF | [80,93 ; 83,12] | **[80,52 ; 83,00]** | idem |
| IC 95 % macro-F1 test, BERT | [75,20 ; 77,63] | **[75,18 ; 77,03]** | idem |
| Macro-F1 validation, BERT | 77,80 | **76,80** | *Books* absent de la validation et jamais prédit par BERT |

Inchangés, vérifiés dans `final_validation.json` → `before_after` :
- scores ponctuels du test, écarts appariés et leurs IC, poids (0,2), seuils, couvertures ;
- recherche (clause → clause et 40 questions) ;
- audit, intégralement.

Le poids d'ensemble a été **refait sur la validation** avec la définition standard et reste 0,2. Ce n'est pas un réglage sur le test.

## Provenance des chiffres

Chaque chiffre du README et du kit de soutenance figure dans [../presentation/key_numbers.md](../presentation/key_numbers.md), avec son fichier et sa clé JSON. Ce fichier est généré par `scripts/key_numbers.py`, sans saisie manuelle. Les macro-F1 de `train_log.json` sont des traces de l'entraînement, calculées avec la définition de l'époque : ils sont présentés comme tels.

## Vérifications de cohérence (`scripts/cross_checks.py --manifest`)

- 100 labels dans le CSV, somme des supports = 10 000 ;
- moyenne des F1 du CSV = macro-F1 du JSON, pour les 3 variantes ;
- configuration, poids et seuils venant de la validation ;
- même ordre des labels pour le dataset, la configuration, BERT et TF-IDF ;
- macro-F1 de validation et de test égal à sklearn `zero_division=0` sur `range(100)`, et définition déclarée dans les deux rapports ;
- 100 labels présents dans le test ;
- empreintes du manifeste égales aux fichiers.

**15/15.**

## Limites restantes

- **Reproductibilité inter-machine non démontrée** : la reconstruction propre a eu lieu sur la même machine, avec les caches uv et Hugging Face partagés. Python 3.14.0b4 est une version bêta.
- **Reprise sur MPS / bfloat16 non démontrée** (testée sur CPU seulement). Elle est sans objet pour le modèle livré, qui n'a pas été repris.
- **Une seule graine** ; les intervalles bootstrap ignorent la corrélation entre clauses d'un même contrat.
- **Recherche** : les 40 questions ont déjà été vues (non-régression) ; le filtre de durée n'est pas sémantique ; la pertinence est approchée par le label.
- **Audit** : groupes par proxys ; petits groupes non concluants.
- **Chemins personnels** : `reports/train_log.json` et les rapports archivés contiennent un chemin absolu avec le nom d'utilisateur (`output_dir`, `model_dir`). Ce sont des traces historiques non réécrites. Si nécessaire, les caviarder dans un commit séparé et explicite.

## Checklist `PROJECT_GUIDELINES.md`

| Exigence | Statut | Où |
|---|---|---|
| Fine-tuner un modèle pré-entraîné (BERT / CamemBERT) | ✅ Legal-BERT (CamemBERT écarté : données en anglais) | `train.py`, README §5 |
| Pipeline complet : tokenisation, boucle d'entraînement, évaluation sur un jeu tenu à part | ✅ | `train.py`, `evaluate.py`, README §6 |
| Classer chaque clause parmi les types de la check-list | ✅ 100 labels, abstention | `inference.py`, CLI `classify` |
| Index de recherche par sentence-transformer | ✅ bge-small + BM25 + RRF | `retrieval.py`, README §7 |
| Recherche par clause ou par question (« 30-day notice ») | ✅ / ⚠ la question vitrine échoue, échec documenté | README §7, `demo.md` |
| Preuves : globales, par type de clause, par type de contrat et de partie | ✅ | `eval_metrics.json`, `per_class_test.csv`, `audit.json` |
| Audit des biais | ✅ par proxys, avec précautions | `audit.py`, README §8 |
| Ce pour quoi le système ne doit pas servir | ✅ | README §11 |
| Résultats reproductibles, évaluation documentée, audit des défaillances | ✅ sur cette machine / ⚠ inter-machine non démontré | manifeste, tests, README §10 |
| Défense des choix devant le CTO | ✅ | `presentation/` |
