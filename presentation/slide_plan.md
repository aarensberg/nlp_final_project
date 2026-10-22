# Plan de soutenance : 12 diapositives (8 à 10 min)

Tous les chiffres viennent de [key_numbers.md](key_numbers.md). Figures : `reports/figures/`.

---

## 1. Le problème métier
- La due diligence demande de lire, typer et comparer des milliers de clauses.
- Trois besoins : **classer** chaque clause, **retrouver** des précédents, **prouver** la fiabilité et les limites.
- Client européen : il faut des résultats reproductibles, auditables, et savoir quand le système ne sait pas.
- **Visuel** : schéma « clause → type + précédents + « à revoir par un juriste » ».
- **Chiffre** : 100 types de clauses.
- **Message oral** : « On n'automatise pas le juriste, on trie et on signale. »

## 2. Données : LEDGAR
- 80 000 clauses de contrats déposés à la SEC (2016–2019), découpage chronologique 60k / 10k / 10k.
- Une étiquette parmi 100 types.
- Anglais, droit américain : **pas européen**.
- **Visuel** : `class_distribution.png`.
- **Chiffre** : *Governing Laws* 3 167 exemples contre *Books* 23 (ratio 138).
- **Message oral** : « Le déséquilibre impose de regarder le macro-F1, pas seulement l'exactitude. »

## 3. Ce que l'exploration a changé
- 282 clauses du test sont identiques à une clause du train : on publie aussi les scores sans ces doublons.
- 11,9 % des clauses dépassent 256 tokens : la troncature est un compromis.
- Pronoms : *he* 2 073 contre *she* 201 ; seulement 253 clauses mentionnent une juridiction hors États-Unis.
- Plusieurs types absents de LEDGAR : RGPD, force majeure, non-concurrence, limitation de responsabilité.
- **Visuel** : `token_lengths.png`.
- **Message oral** : « Ces constats fixent d'avance les limites d'usage. »

## 4. Architecture
- Trois variantes derrière une seule interface : Legal-BERT, TF-IDF + régression logistique, **combinaison 0,2 / 0,8**.
- Calibration par température et **abstention** avec un seuil par variante.
- Recherche : dense (bge-small) + BM25 → fusion RRF, avec un filtre « N jours » optionnel.
- **Visuel** : diagramme de [architecture.md](architecture.md).
- **Message oral** : « Chaque paramètre est choisi sur la validation ; le test sert uniquement à évaluer. »

## 5. Entraînement
- Legal-BERT (anglais juridique) plutôt que CamemBERT (français).
- Boucle PyTorch écrite à la main, 2 époques, 256 tokens, bfloat16 sur un Apple M3 : 146 min.
- Un seul run, sans reprise (provenance documentée) ; reprise testée sur CPU.
- **Chiffre** : macro-F1 de validation 74,0 → 77,6 entre l'époque 1 et l'époque 2 (définition de l'époque).
- **Message oral** : « Le budget est la principale contrainte ; nous en mesurons les effets. »

## 6. Résultats : BERT, TF-IDF, combinaison
- **Combinaison 88,43 / 82,84** (micro / macro-F1 test).
- TF-IDF 87,70 / 82,19 ; BERT seul 86,35 / 76,27.
- Gain sur TF-IDF : **+0,73 point** micro [+0,51 ; +0,98], **+0,65 point** macro [+0,31 ; +1,00].
- **Visuel** : `per_class_f1_vs_frequency.png`.
- **Message oral** : « BERT seul perd contre une baseline forte ; la combinaison apporte un gain modeste mais mesurable. »

## 7. Calibration et abstention
- Seuils choisis sur la validation pour viser 95 % de classifications correctes.
- Sur le test, la combinaison traite 84,2 % des clauses, dont 94,5 % sont correctement classées.
- ECE 0,017 ; 95 % est une cible de validation, **pas une garantie**.
- **Visuel** : `coverage_accuracy.png`, `reliability.png`.
- **Message oral** : « Environ une clause sur six part chez un juriste : c'est voulu. »

## 8. Recherche de précédents
- Clause → clauses similaires : P@1 77,4 %, P@10 66,4 % (même label).
- 40 questions écrites à la main, déjà vues : hybride P@1 92,5 %, non-régression seulement.
- **Échec contrôlé** : « 30-day notice period » donne P@1 = 0 ; le 1er résultat est une clause *Notices* dont les 30 jours ne sont pas un préavis de résiliation.
- **Visuel** : capture `demo_outputs/4_search_30day_controlled_failure.txt`.
- **Message oral** : « Mentionner 30 jours n'est pas exprimer un préavis de 30 jours. »

## 9. Audit
- Clauses de bail : 75,2 % contre 84,3 % attendu ; 87,6 % de correctes parmi les clauses traitées.
- Classes rares : 84,3 % de correctes parmi les clauses traitées.
- Contrefactuels : la juridiction américaine → européenne change le type dans 3,6 % des cas ; bruit OCR 5,4 %.
- Hors périmètre : la combinaison s'abstient sur 8 entrées sur 8 ; BERT seul classe la clause RGPD « General » à 0,90.
- **Visuel** : `slice_accuracy.png`, `counterfactual_flips.png`.
- **Message oral** : « Groupes = proxys textuels ; les petits groupes ne permettent pas de conclure. »

## 10. Limites
- Anglais et SEC uniquement ; droit européen non validé.
- Classes rares, clauses longues, labels quasi synonymes.
- Filtre de durée non sémantique ; une graine, une machine, Python en version bêta.
- **Message oral** : « Nous publions les échecs autant que les réussites. »

## 11. Recommandation de déploiement
- Pilote interne sur clauses anglaises, avec juriste dans la boucle et abstention active.
- Ne jamais utiliser seul pour des décisions juridiques ni sur du droit européen.
- Avant extension : données européennes annotées, types manquants (RGPD, limitation de responsabilité), puis expérience 512 tokens / plus d'époques / plusieurs graines.
- **Message oral** : « Utile pour trier, pas pour décider. »

## 12. Conclusion
- Une chaîne complète, reproductible et auditée : données → modèles → validation → test → recherche → audit.
- Un résultat honnête : la combinaison bat de peu une baseline forte.
- Des limites mesurées et documentées.
- **Chiffre de clôture** : 88,43 % micro-F1, avec 84,2 % des clauses traitées.
- **Message oral** : « La valeur du projet, c'est la confiance qu'on peut lui accorder, et savoir où elle s'arrête. »
