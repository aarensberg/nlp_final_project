# Questions probables du jury et réponses courtes

Chiffres : [key_numbers.md](key_numbers.md). Les références « § » renvoient au [README](../README.md).

**1. Pourquoi Legal-BERT ?**
LEDGAR est en anglais juridique. Legal-BERT a été pré-entraîné sur de la législation, de la jurisprudence et des contrats américains. Dans LexGLUE, c'est l'un des meilleurs modèles « base » sur LEDGAR (88,2 / 83,0), avec un protocole bien plus coûteux que le nôtre.

**2. Pourquoi pas CamemBERT ?**
CamemBERT est un modèle français, et nos données sont anglaises. Le fine-tuner sur LEDGAR ajouterait un décalage de langue sans rien apporter. CamemBERT ne deviendrait pertinent qu'avec un corpus de clauses françaises annoté.

**3. Pourquoi un modèle anglais pour un client européen ?**
Parce que c'est le seul jeu de 100 types de clauses publiquement annoté à cette échelle. Nous l'assumons comme une limite : pas de validation du droit européen, et seulement 253 clauses d'entraînement mentionnent une juridiction hors États-Unis. Le README le classe en usage interdit.

**4. Comment gérez-vous le déséquilibre (138 fois entre la classe la plus fréquente et la plus rare) ?**
Par l'évaluation plutôt que par du rééchantillonnage : macro-F1, scores par classe, groupes tête / milieu / queue, et abstention. Les classes rares restent le point faible : 84,3 % de correctes parmi les clauses traitées.

**5. Micro-F1 ou macro-F1 ?**
Le micro-F1 (ici égal à l'exactitude) pondère chaque clause, le macro-F1 pondère chaque type. Pour un juriste, rater systématiquement un type rare est grave : c'est pour cela que nous suivons le macro-F1.

**6. Quelle définition exacte du macro-F1 ?**
Univers fixe des 100 labels, avec un F1 de 0 pour un label absent, comme sklearn `zero_division=0`. La même définition sert pour les scores ponctuels, le bootstrap et le choix du poids. L'autre convention (un label absent compte 1) n'est publiée qu'en analyse de sensibilité.

**7. Pourquoi tronquer à 256 tokens ?**
À 512, la mémoire partagée de 16 Go saturait et le coût doublait. 256 tokens couvrent 88 % des clauses entières. C'est l'une des deux hypothèses qui expliqueraient le retard de BERT : il perd 3,9 points face à TF-IDF sur les clauses de plus de 200 mots.

**8. BERT est-il sous-entraîné ?**
Nous ne pouvons pas l'affirmer. Nous avons un budget de 2 époques, et le score de validation montait encore (74,0 → 77,6). C'est une hypothèse, à tester avec plus d'époques et plusieurs graines, pas une cause établie.

**9. Comment BERT peut-il perdre contre TF-IDF ?**
Sur LEDGAR, le type d'une clause tient souvent à quelques mots-clés (« governing law », « counterparts »), ce que TF-IDF capte très bien. LexGLUE observe aussi un TF-IDF très proche des transformers sur ce jeu (87,0 contre 88,2). Avec notre petit budget, BERT n'a pas rattrapé cet écart.

**10. À quoi sert la combinaison si le gain n'est que de 0,7 point ?**
Le gain est faible mais réel : sur les mêmes clauses, l'intervalle bootstrap est positif ([+0,51 ; +0,98] en micro-F1). La combinaison est aussi la mieux calibrée (ECE 0,017) et couvre le plus de clauses à cible égale (84,2 %). Sur les 8 entrées hors périmètre testées, elle s'abstient toujours, comme TF-IDF, alors que BERT seul se trompe une fois avec confiance.

**11. Comment avez-vous choisi le poids 0,2 ?**
Par une grille de 0 à 1 (pas de 0,05) sur la **validation**, en retenant le meilleur macro-F1. Le test n'a pas été utilisé. Ce choix est vérifié par un test automatique.

**12. Qu'est-ce que la calibration apporte ?**
Elle rend les scores interprétables comme des fréquences, ce qui est nécessaire pour fixer un seuil. Pour BERT, l'ECE passe de 0,029 à 0,020 avec un scaling par température appris sur la validation.

**13. Comment fonctionne l'abstention ?**
Chaque variante a un seuil, choisi sur la validation : le plus bas dont l'ensemble réellement accepté atteint 95 % de classifications correctes, en prenant les égalités de confiance en entier. Sur le test : 84,2 % de clauses traitées, dont 94,5 % correctes. Ce n'est pas une garantie.

**14. Pourquoi seulement 94,5 % et non 95 % sur le test ?**
Parce que le seuil est appris sur la validation et que le test vient d'une autre année. Un léger écart est normal ; nous ne réajustons pas le seuil sur le test.

**15. Qu'en est-il de la fuite par doublons ?**
282 clauses du test existent dans le train. Sans elles, la combinaison obtient 88,11 / 82,49 contre 88,43 / 82,84 : l'effet est faible, et il est publié.

**16. Vos intervalles de confiance sont-ils fiables ?**
Ce sont des bootstraps par clause. LEDGAR n'a pas d'identifiant de contrat, donc la corrélation entre clauses d'un même contrat est ignorée : les intervalles sont probablement un peu trop étroits. Nous le disons.

**17. Que teste McNemar ?**
La différence de taux d'erreur sur les clauses où les deux systèmes divergent (106 contre 33 entre la combinaison et TF-IDF). Ce n'est pas un test du macro-F1 : pour lui, nous utilisons un bootstrap apparié.

**18. Le baseline est-il trop fort pour être honnête ?**
C'est l'inverse : un baseline fort rend la comparaison crédible. Le nôtre (87,70 / 82,19) est cohérent avec celui publié dans LexGLUE (87,0 / 81,4).

**19. Le système a-t-il des biais ?**
Nous avons audité des groupes approximés par mots-clés. Le défaut le plus net concerne les clauses de bail (−9,2 points par rapport à l'attendu). Sur le genre, le groupe *she* (37 clauses) est trop petit pour conclure. Ce sont des proxys textuels, pas des attributs de personnes.

**20. À quoi servent les contrefactuels ?**
À vérifier que le type ne dépend pas de détails non pertinents. Passer de « he » à « she » change le type dans 2,3 % des cas ; remplacer un État américain par une juridiction européenne, dans 3,6 %. Attention : changer la juridiction change aussi les effets juridiques. Nous ne testons que la stabilité du type.

**21. Comment mesurez-vous la pertinence en recherche ?**
Par le label LEDGAR : un résultat est pertinent s'il a le même type que la requête (protocole A) ou un type accepté (40 questions). C'est une approximation : un résultat du bon type peut ne pas répondre à la question.

**22. Pourquoi la question « 30-day notice » échoue-t-elle ?**
Le filtre de durée vérifie qu'une clause mentionne « 30 days », pas que ce soit un préavis de résiliation. Le 1er résultat est une clause *Notices* dont les 30 jours servent à notifier une « Good Reason ». Nous le montrons volontairement en démonstration. La correction passerait par une extraction structurée (rôle + durée).

**23. Vos 40 questions valident-elles la recherche ?**
Non. Elles ont été vues pendant le développement : elles servent uniquement de non-régression. Une validation demanderait des questions nouvelles, rédigées et jugées par des juristes.

**24. Le projet est-il reproductible ?**
Sur cette machine, oui, et nous l'avons testé : environnement neuf avec `uv sync --frozen`, 92 tests réussis, logits recalculés identiques bit à bit, manifeste avec empreintes. Entre machines, ce n'est pas démontré : précision bfloat16 sur MPS, Python 3.14 en version bêta.

**25. L'entraînement a-t-il été repris en cours de route ?**
Non : le modèle livré vient d'un seul processus, de l'étape 0 à l'étape 7 500. Le `resume: true` du journal enregistre seulement l'option passée (voir `reports/training_provenance.md`). Le mécanisme de reprise est testé sur CPU, mais pas démontré sur MPS.

**26. Quels sont les coûts ?**
Fine-tuning : environ 2 h 30 sur un Apple M3. Baseline : environ 4 min. Index : environ 5 min. Inférence BERT : environ 10 000 clauses en 2 min sur le même GPU. Une commande `classify` isolée prend environ 10 s, surtout pour charger le modèle.

**27. Et la confidentialité des contrats clients ?**
Tout tourne en local, sans API externe : les modèles sont téléchargés une fois, avec des révisions figées. Les contrats clients ne doivent jamais être versionnés, et `artifacts/` est exclu de Git. Le manifeste ne contient aucun chemin personnel. Un déploiement réel demanderait en plus une revue sécurité et RGPD.

**28. Quels usages sont interdits ?**
Décider sans juriste. Le droit européen. Les langues autres que l'anglais. La détection exhaustive des risques. Les types absents de LEDGAR (RGPD, force majeure, limitation de responsabilité…). La recherche d'un préavis précis. Les conclusions d'équité sur des personnes.

**29. Que feriez-vous avec deux semaines de plus ?**
Une expérience contrôlée (256 contre 512 tokens, 2 époques contre plus avec arrêt anticipé, 3 graines). Une fusion des labels quasi synonymes, validée par des juristes. Des clauses européennes annotées. Et une extraction structurée des délais.
