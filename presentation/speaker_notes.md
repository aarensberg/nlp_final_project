# Script oral (8 à 10 min)

Les temps sont indicatifs. Chiffres : [key_numbers.md](key_numbers.md).

**[0:00, slide 1, problème]**
Dans une due diligence, un juriste lit chaque clause, lui donne un type, cherche des précédents et signale ce qui est inhabituel. Notre cœur NLP fait trois choses : classer chaque clause parmi 100 types, retrouver des clauses proches, et dire quand il ne faut pas lui faire confiance. Ce troisième point compte autant que les deux premiers, parce que nos clients européens veulent un système auditable, pas une boîte noire.

*Transition : avant de modéliser, regardons les données.*

**[0:50, slide 2, LEDGAR]**
Nous utilisons LEDGAR : 80 000 clauses de contrats américains déposés à la SEC, découpées dans le temps. Le train couvre 2016–2017, le test 2019. La première contrainte est le déséquilibre : 3 167 exemples de *Governing Laws* contre 23 de *Books*. Une exactitude globale masquerait les classes rares. Nous rapportons donc le macro-F1, qui donne le même poids à chaque type.

**[1:40, slide 3, exploration]**
L'exploration a changé nos choix. 282 clauses du test existent mot pour mot dans le train : nous publions aussi les scores sans elles. Environ 12 % des clauses dépassent 256 tokens, ce qui fera de la troncature un compromis. Et plusieurs types utiles en due diligence n'existent pas dans LEDGAR : RGPD, force majeure, non-concurrence, limitation de responsabilité. Le système ne peut donc pas les reconnaître.

*Transition : voici ce que nous avons construit.*

**[2:30, slide 4, architecture]**
Trois variantes partagent une seule interface. Legal-BERT, fine-tuné. Un baseline TF-IDF avec régression logistique. Et une combinaison des deux, avec 20 % pour BERT et 80 % pour TF-IDF. Chaque sortie est calibrée, puis comparée à un seuil : sous le seuil, le système s'abstient et renvoie la clause à un juriste. Point important : températures, poids et seuils ont tous été choisis sur la validation. Le test n'a servi qu'à mesurer.

**[3:20, slide 5, entraînement]**
Pourquoi Legal-BERT et pas CamemBERT ? Les données sont en anglais juridique américain, et Legal-BERT a été pré-entraîné sur ce type de textes. Nous avons écrit la boucle d'entraînement nous-mêmes et entraîné 2 époques à 256 tokens sur un portable : 146 minutes. Le modèle livré vient d'un seul run, sans reprise, et nous l'avons documenté.

**[4:00, slide 6, résultats]**
Le résultat principal est inconfortable, et nous le présentons tel quel : **Legal-BERT seul fait moins bien que le baseline TF-IDF**, 86,35 contre 87,70 en micro-F1, et surtout 76,27 contre 82,19 en macro-F1. Nous avons deux hypothèses, pas une preuve : un budget de deux époques, alors que le score de validation montait encore, et la troncature, puisque BERT perd surtout sur les clauses longues. La combinaison atteint 88,43 / 82,84. Le gain sur TF-IDF est modeste, +0,7 point, mais il est mesuré sur les mêmes clauses et son intervalle de confiance reste positif. En pratique, BERT apporte un signal complémentaire, pas une révolution.

*Transition : un score moyen ne suffit pas pour décider, il faut savoir quand s'abstenir.*

**[5:10, slide 7, abstention]**
Nous avons fixé un seuil sur la validation pour viser 95 % de classifications correctes. Sur le test, la combinaison traite 84 % des clauses, dont 94,5 % sont correctement classées, et laisse environ une clause sur six à un juriste. Ces 95 % sont une cible de validation, pas une garantie, encore moins sur des contrats d'un autre pays.

**[6:00, slide 8, recherche]**
Pour la recherche, nous combinons des embeddings et BM25. Sur 981 clauses du test, le premier résultat a le bon type dans 77 % des cas. Nous montrons aussi un échec volontairement : la question « termination clauses with a 30-day notice period » renvoie d'abord une clause qui mentionne 30 jours… pour notifier autre chose. Notre filtre vérifie qu'une durée est mentionnée, pas qu'elle joue le rôle de préavis. Nos 40 questions de test ont été vues pendant le développement : ce sont des tests de non-régression, pas une validation.

**[7:00, slide 9, audit]**
L'audit fait ressortir deux faiblesses nettes : les clauses de bail, environ 9 points sous ce que leurs types laissent attendre, et les classes rares, où même après abstention seulement 84 % des réponses sont correctes. Remplacer une juridiction américaine par une européenne change le type prédit dans 3,6 % des cas. Sur des entrées hors sujet, la combinaison s'abstient toujours ; BERT seul, lui, classe une clause RGPD avec 90 % de confiance. Nos groupes sont des proxys par mots-clés, donc nous ne tirons aucune conclusion d'équité sur les petits groupes.

**[8:00, slides 10 et 11, limites et déploiement]**
Pourquoi ce système ne remplace pas un avocat : il connaît des types de clauses, pas leurs effets juridiques. Il ignore le droit européen et des types essentiels, et il se trompe surtout là où un juriste est le plus utile, sur les cas rares. Nous recommandons donc un pilote interne sur des clauses anglaises, avec un juriste dans la boucle et l'abstention activée. Avant d'aller plus loin, il faut des données européennes annotées et une expérience contrôlée sur la longueur et le nombre d'époques.

**[9:00, slide 12, conclusion]**
Nous livrons une chaîne complète et reproductible, un résultat honnête où la combinaison bat de peu une baseline forte, et une cartographie précise de ce que le système ne sait pas faire. Merci.
