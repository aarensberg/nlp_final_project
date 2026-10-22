# Script de soutenance à trois voix — Alessandro · Charles · Louis

Support associé : [slides/index.html](slides/index.html) (18 diapositives + 6 annexes pour les questions ; 11 animations Manim).
Commandes du support : ← → naviguer · **N** notes orateur · **F** plein écran · **R** rejouer l'animation · **S** images fixes (plan B si une vidéo ne se lit pas).
Durée visée : **12 à 13 min** de présentation, puis questions ([jury_qa.md](jury_qa.md)).
Chiffres : tous repris de [key_numbers.md](key_numbers.md), lui-même généré depuis `reports/` ; ils ont été revérifiés dans `reports/eval_metrics.json`, `reports/retrieval_metrics.json`, `reports/audit.json` et `presentation/demo_outputs/` avant la rédaction.

---

## Partie A — Ce que l'analyse du dépôt permet de défendre

| Statut | Contenu |
|---|---|
| **Implémenté et validé par des tests** | 3 variantes derrière une même interface (`inference.py`) ; boucle d'entraînement PyTorch écrite à la main ; baseline TF-IDF ; calibration par température ; abstention à seuil ; index dense + BM25 + RRF avec filtre « N jours » ; audit (groupes, contrefactuels, hors périmètre) ; manifeste avec empreintes ; 92 tests réussis, 15/15 vérifications croisées |
| **Mesuré (test LEDGAR, consulté une fois)** | combinaison 88,43 / 82,84 (micro / macro-F1) ; TF-IDF 87,70 / 82,19 ; Legal-BERT 86,35 / 76,27 ; gain combinaison − TF-IDF +0,73 pt [+0,51 ; +0,98] ; couverture 84,2 % avec 94,5 % de correctes parmi les clauses traitées ; recherche P@1 77,4 %, P@10 66,4 % |
| **Seulement interprété (hypothèses)** | pourquoi BERT est sous TF-IDF (2 époques, troncature 256) ; pertinence de recherche approchée par le label ; groupes d'audit = proxys par mots-clés |
| **Non démontré** | reproductibilité inter-machine ; variance entre graines ; reprise d'entraînement identique sur MPS ; validité juridique de quoi que ce soit |

**Messages-clés de la soutenance**
1. On ne remplace pas le juriste : on **trie**, on **retrouve**, on **signale** quand on ne sait pas.
2. Résultat honnête : Legal-BERT seul **perd** contre un baseline fort ; la combinaison gagne **peu**, mais de façon mesurée.
3. L'abstention est un outil de tri calibré sur la validation, **pas une garantie**.
4. La recherche marche bien quand la formulation est proche des clauses, et **échoue de façon documentée** sur « 30-day notice ».
5. L'audit cartographie les faiblesses : bail, classes rares, labels quasi synonymes, hors périmètre.
6. Tout est reproductible **sur cette machine**, avec empreintes ; pas au-delà.

**Limites à assumer à l'oral (sans s'excuser)** : anglais / SEC uniquement ; types RGPD, force majeure, non-concurrence, limitation de responsabilité absents ; une graine, une machine, Python bêta ; intervalles bootstrap probablement un peu trop étroits (pas d'identifiant de contrat) ; 40 questions déjà vues = non-régression.

---

## Partie B — Plan de présentation (18 diapositives)

| # | Section | Diapositive | Intervenant | Durée | Visuel |
|---|---|---|---|---|---|
| 1 | Ouverture | Titre | Alessandro | 0:20 | — |
| 2 | Problème | Le besoin métier | Alessandro | 0:50 | Manim : juriste → assistant → 3 sorties |
| 3 | Données | Pourquoi LEDGAR | Alessandro | 0:45 | Manim : découpage chronologique |
| 4 | Données | Les limites de LEDGAR | Alessandro | 0:50 | Manim : déséquilibre + types absents |
| 5 | Architecture | Vue d'ensemble | Alessandro → Charles | 0:50 | Manim : architecture complète |
| 6 | Classification | Baseline TF-IDF | Charles | 0:35 | — |
| 7 | Classification | Fine-tuning Legal-BERT | Charles | 0:50 | — |
| 8 | Résultats | Trois variantes comparées | Charles | 1:00 | Manim : barres + IC 95 % |
| 9 | Classification | La combinaison | Charles | 0:45 | Manim : formule + 2 exemples réels |
| 10 | Fiabilité | Calibration et abstention | Charles → Louis | 0:55 | Manim : 100 clauses (courbe de couverture en annexe A1) |
| 11 | Recherche | Dense + BM25 + RRF | Louis | 0:50 | Manim : pipeline de recherche |
| 12 | Recherche | Exemple réussi | Louis | 0:30 | sortie réelle |
| 13 | Recherche | Échec contrôlé « 30-day notice » | Louis | 0:50 | Manim : surlignage du faux positif |
| 14 | Audit | Groupes, contrefactuels, hors périmètre | Louis | 1:05 | Manim : groupes → contrefactuels → hors périmètre (figure complète en annexe A4) |
| 15 | Reproductibilité | Chaîne d'empreintes | Louis → Alessandro | 0:40 | Manim : chaîne de traçabilité |
| 16 | Usages | Autorisés / interdits | Alessandro | 0:40 | — |
| 17 | Limites | Limites et suite | Alessandro | 0:40 | — |
| 18 | Conclusion | Recommandation | Alessandro | 0:35 | — |

**Logique de répartition.**
- **Alessandro** pose le cadre (problème, données, architecture), puis revient pour **la décision** : usages, limites, recommandation. Le jury entend la même voix au début et à la fin.
- **Charles** porte tout ce qui touche aux **modèles de classification** : TF-IDF, Legal-BERT, résultats, combinaison, abstention.
- **Louis** porte la **recherche** et la **preuve de confiance** : exemple, échec contrôlé, audit, reproductibilité.

**Si le temps est réduit à 10 min** : fusionner 6 et 7 (Charles), raccourcir 12 à une phrase (Louis), fusionner 16 et 17 (Alessandro).

---

## Partie C — Script complet

Conventions : **[S#]** = diapositive ; ⚑ = point à appuyer à l'oral ; ⚠ = précaution de formulation. Les animations Manim démarrent seules à l'arrivée sur la diapositive (clic sur la vidéo pour la rejouer).

---

### [S1] Titre — **Alessandro** (0:20)

*Rôle de la slide : poser le sujet et l'équipe.*

> Bonjour. Nous allons vous présenter le cœur NLP d'un assistant de revue de contrats : il classe les clauses, retrouve des précédents dans une banque de clauses, et documente où on peut lui faire confiance — et où on ne peut pas. Je suis Alessandro, je vais poser le problème et les données ; Charles présentera les modèles de classification ; Louis la recherche, l'audit et la reproductibilité. Je reviendrai pour la recommandation finale.

⚠ Dire « prototype de recherche » dès que l'occasion se présente, pas « produit ».

---

### [S2] Le besoin métier — **Alessandro** (0:50)

*Rôle : faire comprendre le besoin réel avant toute technique. Animation : un juriste lit des clauses une par une ; l'assistant produit trois sorties.*

> Pendant une due diligence — un rachat, un financement, un audit fournisseurs — un juriste junior lit chaque clause, lui donne un type, cherche des précédents dans la banque du cabinet, et signale ce qui est inhabituel à un associé. C'est des milliers d'heures.
>
> Notre cœur NLP doit faire trois choses. Un : **classer** chaque clause parmi 100 types. Deux : **retrouver** des clauses similaires, à partir d'une clause collée ou d'une question. Trois : **prouver** qu'il fonctionne, et dire quand il ne faut pas lui faire confiance.
>
> Ce troisième point n'est pas un bonus : nos clients sont européens, et ils ne déploieront pas une boîte noire. Ils veulent des résultats reproductibles et un audit des défaillances.

⚑ « On n'automatise pas le juriste : on trie, on retrouve, on signale. »
⚠ Ne pas dire que le système « comprend » ou « analyse juridiquement » les contrats.

---

### [S3] Pourquoi LEDGAR — **Alessandro** (0:45)

*Rôle : justifier le choix du jeu de données. Animation : frise 2016 → 2019 avec les trois découpages.*

> Nous travaillons sur LEDGAR, le sous-ensemble de LexGLUE dédié aux clauses. 80 000 clauses tirées de contrats déposés à la SEC, chacune avec **une** étiquette parmi **100 types**.
>
> Deux raisons à ce choix. C'est le seul jeu public annoté à cette échelle pour la classification de clauses. Et il a un benchmark publié, LexGLUE, qui nous donne un point de comparaison.
>
> Le découpage est chronologique : 60 000 clauses de 2016–2017 pour l'entraînement, 10 000 de 2018 pour la validation, 10 000 de 2019 pour le test. Le test mesure donc déjà un peu de dérive dans le temps. Et une règle que nous avons tenue partout : **tous les réglages se font sur la validation, le test ne sert qu'à mesurer.**

⚑ La règle validation / test : on la répétera, c'est la colonne vertébrale de la crédibilité.

---

### [S4] Les limites de LEDGAR — **Alessandro** (0:50)

*Rôle : poser les limites dès les données, pas à la fin. Animation : barre « Governing Laws 3 167 » contre « Books 23 », puis liste des types absents.*

> LEDGAR a trois limites que nous avons mesurées avant de modéliser.
>
> D'abord le **déséquilibre** : 3 167 exemples de *Governing Laws* contre 23 de *Books*, un rapport de 138. Une exactitude globale masquerait les types rares : nous suivons donc le **macro-F1**, qui donne le même poids à chaque type.
>
> Ensuite le **périmètre** : c'est de l'anglais et du droit américain. Seules 253 clauses d'entraînement mentionnent une juridiction hors des États-Unis.
>
> Enfin, des **types absents** de la taxonomie, et pourtant centraux en due diligence européenne : RGPD, force majeure, non-concurrence, limitation de responsabilité. Le système ne peut pas les reconnaître : il n'y a pas d'étiquette pour eux.
>
> Un dernier point de propreté : 282 clauses du test existent mot pour mot dans l'entraînement. Nous publions aussi les scores sans elles ; l'écart est faible.

⚠ Ne pas présenter les 253 clauses comme « européennes » : c'est « hors États-Unis », détecté par mots-clés.

*Transition* → « Voici comment nous avons construit le système autour de ces contraintes. »

---

### [S5] Architecture — **Alessandro**, puis passage à **Charles** (0:50)

*Rôle : donner la carte complète avant le détail. Animation : deux branches de classification fusionnées, seuil, puis branche recherche.*

> Le système a deux chaînes.
>
> En haut, la **classification** : une clause passe par deux modèles — Legal-BERT fine-tuné et un TF-IDF avec régression logistique. Leurs scores sont calibrés, puis combinés avec un poids de 0,2 pour BERT et 0,8 pour TF-IDF. Si le score maximum dépasse un seuil, on renvoie le type ; sinon, la clause part « à revoir par un juriste ».
>
> En bas, la **recherche** : un filtre optionnel sur les durées, puis deux moteurs, un dense et un lexical, fusionnés par RRF.
>
> Les trois variantes de classification passent par la même interface, et toutes sont rapportées : on ne cache pas la moins bonne. Charles va vous détailler la classification.

**Charles** (enchaîne) : « Merci. Je commence par le modèle le plus simple, parce qu'il s'est révélé le plus difficile à battre. »

---

### [S6] Baseline TF-IDF — **Charles** (0:35)

*Rôle : montrer que le baseline est sérieux.*

> Le baseline, c'est un TF-IDF sur les unigrammes et bigrammes, suivi d'une régression logistique. Environ 4 minutes d'entraînement sur CPU.
>
> Pourquoi s'en soucier ? Parce que sur LEDGAR, le type d'une clause tient souvent à quelques mots-clés : « governing law », « counterparts », « severability ». Un modèle lexical capte cela très bien. Notre baseline obtient 87,70 en micro-F1 et 82,19 en macro-F1 sur le test, du même ordre que celui publié dans LexGLUE, 87,0 / 81,4.

⚑ Un baseline fort rend la comparaison crédible.
⚠ La comparaison LexGLUE est **indicative** : même test, protocole différent (SVM, grille de recherche).

---

### [S7] Fine-tuning Legal-BERT — **Charles** (0:50)

*Rôle : expliquer le choix du modèle et le budget réel.*

> Pour le modèle neuronal, nous avons choisi **Legal-BERT** plutôt que CamemBERT. CamemBERT est un modèle français, nos données sont en anglais juridique américain ; Legal-BERT a justement été pré-entraîné sur ce type de textes, notamment des contrats américains.
>
> Nous avons écrit la boucle d'entraînement nous-mêmes en PyTorch : lots groupés par longueur, AdamW avec warm-up, taux d'apprentissage 3·10⁻⁵, checkpoints reprenables. Le budget est celui d'un portable : un Apple M3, **2 époques**, des clauses tronquées à **256 tokens**, ce qui couvre 88 % des clauses entières. À 512, la mémoire saturait. Résultat : un seul run de 146 minutes, sans reprise — c'est documenté dans un fichier de provenance.
>
> Sur la validation, le macro-F1 passait de 74,0 à 77,6 entre la première et la deuxième époque. Il montait encore.

⚑ « Le budget est notre principale contrainte, et nous en mesurons les effets. »
⚠ Les valeurs 74,0 → 77,6 utilisent l'ancienne définition du macro-F1 (labels présents) : ce sont des traces historiques. Si le jury demande, le dire.

---

### [S8] Résultats comparés — **Charles** (1:00)

*Rôle : le résultat principal, sans le maquiller. Animation : barres micro et macro-F1 avec intervalles de confiance 95 %.*

> Voici les résultats sur les 10 000 clauses du test.
>
> Le résultat principal est inconfortable, et nous le présentons tel quel : **Legal-BERT seul fait moins bien que TF-IDF**. 86,35 contre 87,70 en micro-F1. Et surtout 76,27 contre 82,19 en macro-F1 : BERT décroche sur les types rares — 61,9 en macro-F1 sur les classes de queue, contre 74,3 pour TF-IDF.
>
> Pourquoi ? Nous avons deux **hypothèses**, pas une preuve. Le budget de deux époques, alors que le score montait encore. Et la troncature : sur les clauses de plus de 200 mots, BERT perd 3,9 points d'exactitude face à TF-IDF, contre 0,3 point sur les clauses courtes. Il faudrait une expérience contrôlée, sur plusieurs graines, pour trancher.
>
> La combinaison, elle, atteint **88,43 en micro-F1 et 82,84 en macro-F1**.

⚑ Montrer les intervalles : ceux de BERT ne recouvrent pas ceux de TF-IDF en macro-F1.
⚠ Ne jamais dire « BERT est sous-entraîné » comme un fait. Dire « hypothèse ».
⚠ Les IC sont des bootstraps par clause ; sans identifiant de contrat, ils sont probablement un peu trop étroits.

---

### [S9] La combinaison — **Charles** (0:45)

*Rôle : expliquer la fusion et sa valeur réelle. Animation : formule de combinaison, puis deux sorties réelles enregistrées.*

> La combinaison est une moyenne pondérée des probabilités calibrées : 0,2 pour BERT, 0,8 pour TF-IDF. Le poids a été choisi par une grille de 0 à 1, par pas de 0,05, sur le meilleur macro-F1 **de validation**.
>
> Le gain sur TF-IDF est **réel mais modeste** : +0,73 point de micro-F1, avec un intervalle apparié de +0,51 à +0,98, et +0,65 point de macro-F1. Sur les clauses où les deux systèmes divergent, la combinaison a raison 106 fois et TF-IDF 33 fois.
>
> Concrètement, BERT apporte un signal complémentaire. Pas une révolution.

⚑ « Faible mais mesuré sur les mêmes clauses, et l'intervalle reste positif. »
⚠ McNemar teste le taux d'erreur, **pas** le macro-F1. Le gain macro est appuyé par un bootstrap apparié.

---

### [S10] Calibration et abstention — **Charles**, puis passage à **Louis** (0:55)

*Rôle : montrer comment le système sait dire « je ne sais pas ». Animation : 100 carrés (≈ 1 % chacun), 84 traités, 16 renvoyés au juriste, ≈ 5 erreurs malgré tout (84,2 % × 5,5 % ≈ 4,6 %, calcul à partir des deux chiffres du rapport).*

> Un score moyen ne suffit pas : en pratique, il faut savoir **quand ne pas répondre**.
>
> Chaque modèle est d'abord calibré par une température apprise sur la validation : ses scores deviennent interprétables comme des fréquences. L'erreur de calibration de la combinaison est de 0,017 sur le test.
>
> Ensuite, nous avons fixé sur la validation le seuil le plus bas qui atteint 95 % de classifications correctes. Pour la combinaison, ce seuil vaut 0,709. Sur le test, elle traite 84,2 % des clauses, et parmi celles-ci, 94,5 % sont correctement classées. Environ **une clause sur six part chez un juriste : c'est voulu**.
>
> Attention : 95 % est une **cible de validation**, pas une garantie. Sur le test, d'une autre année, on obtient 94,5 %. Et sur des contrats d'un autre pays, on ne sait pas. Louis va maintenant vous montrer la recherche de précédents.

⚠ Ne jamais dire « le système garantit 95 % ». Dire « vise », « sur la validation ».
⚠ La confiance porte sur le **type** de clause, pas sur la validité juridique : la CLI l'écrit elle-même.

---

### [S11] Recherche : dense + BM25 + RRF — **Louis** (0:50)

*Rôle : expliquer la recherche hybride. Animation : requête → filtre → deux moteurs → fusion → top-k.*

> Deuxième besoin : retrouver des précédents. La banque contient 59 080 clauses d'entraînement, sans doublons exacts.
>
> Nous combinons deux moteurs. Un moteur **dense**, avec l'encodeur bge-small, qui capte la proximité de sens. Et **BM25**, qui capte les mots exacts. Les deux classements sont fusionnés par **Reciprocal Rank Fusion** : chaque clause reçoit 1 / (60 + son rang) dans chaque liste, et on additionne.
>
> En amont, un filtre optionnel garde seulement les clauses qui mentionnent une durée, par exemple « 30 jours ».
>
> Sur 981 clauses du test utilisées comme requêtes, le premier résultat a le même type dans 77,4 % des cas, et 66,4 % des 10 premiers. L'hybride fait mieux que chaque moteur seul, de peu.

⚠ La pertinence est approchée par **le label LEDGAR** : un résultat du bon type peut ne pas répondre à la question.
⚠ Les 40 questions écrites à la main ont **déjà été vues** : non-régression, pas validation (hybride P@1 92,5 %).

---

### [S12] Exemple réussi — **Louis** (0:30)

*Rôle : montrer le cas favorable, sortie réelle enregistrée.*

> Quand la question ressemble à une clause, cela marche bien. Requête : « either party may terminate on thirty days written notice ». Le filtre retient 824 clauses admissibles, et les 5 premiers résultats sont tous des clauses de résiliation avec un préavis de trente jours.

⚠ Dire : « sortie réelle enregistrée » si on ne fait pas la démo en direct.
⚑ Préparer la bascule : « Mais regardez ce qui se passe si on reformule. »

---

### [S13] Échec contrôlé : « 30-day notice » — **Louis** (0:50)

*Rôle : montrer volontairement un échec et l'expliquer. Animation : le premier résultat est une clause Notices, « 30 days » surligné, mais le rôle de la durée n'est pas un préavis de résiliation.*

> Même intention, formulation de juriste : « show me termination clauses with a 30-day notice period ».
>
> Le premier résultat est une clause *Notices*. Elle contient bien « 30 days »… mais ce sont les 30 jours dont dispose un salarié pour notifier une « Good Reason ». Ce n'est pas un préavis de résiliation. Le premier résultat de type résiliation arrive en 4ᵉ position, et un seul des 10 premiers est pertinent.
>
> La cause est claire : notre filtre vérifie qu'une clause **mentionne** N jours, pas que cette durée **joue le rôle** de préavis. Il ne distingue pas non plus jours ouvrés et calendaires. Le système l'écrit lui-même dans sa sortie. La correction passerait par une extraction structurée : rôle de la durée, valeur, type de jours.

⚑ « Mentionner 30 jours, ce n'est pas exprimer un préavis de 30 jours. »
⚠ Ne pas dire que la recherche « comprend » la notion de préavis, même pour l'exemple réussi.

*Transition* → « Cet échec, nous l'avons trouvé parce que nous l'avons cherché. C'est le rôle de l'audit. »

---

### [S14] Audit — **Louis** (1:05)

*Rôle : cartographier où le système est fragile. Animation en trois temps : groupes (bail, classes de queue), contrefactuels, hors périmètre.*

> L'audit a trois volets.
>
> **Les groupes.** Nous découpons le test par famille de contrats, juridiction, pronoms, longueur, fréquence de classe. Le défaut le plus net : les **clauses de bail**, 266 clauses, 75,2 % d'exactitude alors que leur mélange de types laisserait attendre 84,3 %. C'est présent dans les trois variantes. Et sur les **classes rares**, même après abstention, seules 84,3 % des réponses sont correctes : l'abstention ne suffit pas.
>
> **Les contrefactuels.** On modifie un détail et on regarde si le type change. Inverser « he » et « she » : 2,3 % de changements. Remplacer un État américain par une juridiction européenne : 3,6 %. Un bruit de type OCR : 5,4 %.
>
> **Le hors périmètre.** Huit entrées sans étiquette valide : RGPD, force majeure, français, allemand… La combinaison s'abstient sur les huit. Mais BERT seul classe la clause RGPD en « General » avec 0,90 de confiance. Un seuil **réduit** le risque, il ne le supprime pas.

⚠ Les groupes sont des **proxys par mots-clés**, pas des attributs de personnes. Le groupe « she » (37 clauses) et le groupe hors États-Unis (36) sont **trop petits pour conclure**.
⚠ Changer une juridiction change aussi le contenu juridique : on teste la stabilité du **type**, pas l'équivalence juridique.
⚠ « 8 sur 8 » est un petit échantillon : ne pas en faire une garantie.

---

### [S15] Reproductibilité — **Louis**, puis passage à **Alessandro** (0:40)

*Rôle : montrer que les chiffres sont traçables. Animation : chaîne uv.lock → révisions figées → artefacts empreintés → manifeste → tests.*

> Dernier point : peut-on refaire nos chiffres ? Les dépendances sont verrouillées dans `uv.lock`, les révisions du jeu de données et des modèles sont figées. Chaque artefact est lié à son empreinte : si un poids ou un index change, il est refusé ou reconstruit. Un manifeste enregistre le commit, les versions et les empreintes de tous les rapports.
>
> Sur une copie propre, 92 tests passent, et les logits recalculés sont identiques bit à bit.
>
> Ce que nous ne démontrons pas : la reproductibilité sur une autre machine, la variance entre graines — nous n'avons qu'un seul run — et nous utilisons une version bêta de Python. Alessandro va conclure.

⚠ « Reproductible sur cette machine », jamais « reproductible partout ».

---

### [S16] Usages autorisés / interdits — **Alessandro** (0:40)

*Rôle : traduire l'audit en règles d'usage.*

> Qu'est-ce que cela veut dire pour un client ?
>
> **Usage prévu** : aide au tri de clauses **anglaises** de type SEC, avec relecture humaine systématique, et les clauses sous le seuil envoyées à un juriste.
>
> **À ne pas faire** : décider seul du type ou de la portée d'une clause ; l'utiliser sur du droit européen ou en français, en allemand ; s'en servir pour détecter exhaustivement les risques — il classe des types, il ne juge pas la conformité ; chercher les types qu'il ne connaît pas, comme le RGPD ou la limitation de responsabilité ; retrouver un préavis précis ; et tirer des conclusions d'équité sur des personnes.

⚑ « Classer un type de clause n'est pas juger sa validité. »

---

### [S17] Limites et suite — **Alessandro** (0:40)

*Rôle : assumer les limites et montrer qu'on sait comment avancer.*

> Nos limites principales : BERT seul sous TF-IDF et un gain de combinaison d'environ 0,7 point ; des labels quasi synonymes — *Applicable Laws* et *Governing Laws*, *Waivers* et *No Waivers* — qui plafonnent les scores ; le bail et les classes rares ; une recherche qui ne comprend pas la notion de préavis ; une graine, une machine.
>
> La suite, dans l'ordre : une expérience contrôlée — 256 contre 512 tokens, 2 époques contre davantage avec arrêt anticipé, plusieurs graines, même protocole ; des clauses européennes annotées avec les types manquants ; une extraction structurée des délais ; et une fusion des labels quasi synonymes, validée par des juristes.

---

### [S18] Conclusion — **Alessandro** (0:35)

*Rôle : message final et recommandation.*

> Pour conclure. Nous livrons une chaîne complète — données, modèles, validation, test, recherche, audit — reproductible sur notre machine. Un résultat honnête : la combinaison atteint 88,43 % de micro-F1 et bat de peu un baseline fort. Et une cartographie précise de ce que le système ne sait pas faire.
>
> Notre recommandation : un **pilote interne**, sur des clauses anglaises, avec un juriste dans la boucle et l'abstention activée. **Utile pour trier, pas pour décider.**
>
> Merci. Nous sommes prêts pour vos questions.

---

## Partie D — Répartition des questions du jury

| Thème | Qui répond | Références |
|---|---|---|
| Données, LEDGAR, choix européens, usages | Alessandro | [jury_qa.md](jury_qa.md) Q3, Q15, Q28 |
| TF-IDF, BERT, macro-F1, combinaison, calibration, abstention | Charles | Q1–Q2, Q4–Q14, Q16–Q18 |
| Recherche, audit, reproductibilité, provenance, coûts, confidentialité | Louis | Q19–Q27 |
| « Et avec deux semaines de plus ? » | Alessandro | Q29 |

## Partie E — Annexes du support (questions du jury)

A1 couverture / exactitude · A2 calibration · A3 F1 par classe selon la fréquence · A4 exactitude par groupe · A5 contrefactuels (3 variantes) · A6 distribution des classes et longueurs. Ce sont les figures de `reports/figures/`, inchangées.

## Partie F — Checklist de répétition

- [ ] Chacun relit ses ⚠ : aucun « garantit », « comprend », « prêt pour la production ».
- [ ] Chronométrer un passage complet ; viser 12–13 min.
- [ ] Vérifier les transitions entre intervenants (S5, S10, S15).
- [ ] Si démo en direct : suivre [demo.md](demo.md) ; sinon dire « sortie enregistrée » aux S12–S13.
