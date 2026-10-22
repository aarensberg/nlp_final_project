# Provenance de l'entraînement Legal-BERT livré

Ce document distingue ce qui est **prouvé par des fichiers**, ce qui n'est qu'une **trace déclarative** et ce qui reste **inconnu**. Aucun fichier de trace n'a été modifié a posteriori.

## Faits vérifiables

| Fait | Preuve |
|---|---|
| Le modèle livré a été produit par un seul processus, parti de l'étape 0 et arrivé à l'étape 7 500 (2 époques × 3 750 étapes) | `artifacts/train.log` (non versionné) : la progression commence à `epoch 1/2: 0/3750` ; aucune ligne `resumed at …` |
| Durée d'entraînement : 146,5 min, cohérente avec une exécution unique | `reports/train_log.json` → `train_minutes = 146.497` ; checkpoint final `state.elapsed = 8787.99 s` |
| Scores de validation : époque 1 → 84,18 / 74,02 ; époque 2 → 86,37 / 77,58 (micro / macro-F1, **ancienne** définition sklearn implicite) | `reports/train_log.json` → `epochs[*]` ; identiques dans le checkpoint |
| Époque retenue : 2 | `reports/train_log.json` → `best_epoch` |
| Poids écrits à la fin du run (21/10/2026, 23:33:46, heure locale) | date de modification de `artifacts/classifier/model.safetensors` |
| Le checkpoint final ne contient ni configuration ni états aléatoires | clés de `artifacts/classifier_checkpoint/state.pt` : `epochs, model, optim, sched, state` (écrit avant l'ajout de ces champs) |
| Appareil : `mps` (GPU intégré Apple M3), bfloat16 autocast | `reports/train_log.json` → `device`, `config.amp` |

## Trace déclarative à interpréter avec prudence

- `reports/train_log.json` contient `config.resume = true`. Ce champ enregistre seulement **l'option passée** (`train --resume`), pas une reprise effective. À ce moment, aucun checkpoint n'existait pour ce dossier de sortie. Le code de l'époque repartait alors **silencieusement** de zéro. Ce comportement est maintenant interdit : `--resume` sans checkpoint échoue, voir `train.py`.

## Historique de la session (non vérifiable par fichier)

Ces éléments viennent du journal de travail ; leurs fichiers de trace ont été écrasés, car `train.log` est réécrit à chaque lancement :
1. Un essai de fumée à 512 tokens a été abandonné à cause d'une saturation mémoire.
2. Une première tentative complète (3 époques, avant l'ajout des checkpoints) a été arrêtée par l'utilisateur pendant la 1ʳᵉ époque, **sans checkpoint**.
3. Le run livré a été lancé ensuite (2 époques, 256 tokens, `--resume`) et s'est terminé sans interruption.

## Limites

- **Pas d'identité numérique démontrée sur MPS / bfloat16.** L'équivalence « interrompu puis repris = continu » est seulement testée sur CPU avec un mini-modèle (`tests/test_resume.py`). Elle est sans objet pour le run livré, qui n'a pas été repris.
- La sélection de l'époque a utilisé l'ancienne définition du macro-F1 (moyenne sur les labels présents). Avec une seule classe absente de la validation (*Books*), un changement de définition ne peut pas inverser l'ordre des deux époques. L'écart de 3,6 points dépasse largement l'effet maximal d'un label sur 100 (1 point).
- Une seule graine (42) et un seul run : la variance entre graines n'est pas mesurée.
