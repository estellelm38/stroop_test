# Projet Cognition & Apprentissage : Modélisation Effet Stroop

## 1. Contexte
Dans le cadre d’un projet en sciences cognitives, deux étudiants en L3 MIASHS on décidé
d’étudier l’effet Stroop avec un modèle computationnel cognitif.

## 2. Méthodologie
Cette section détaille la méthodologie utilisée pour collecter les données et effectuer
l'analyse de régression linéaire multiple.

### 2.1 Collecte des données expérimentales
Dans cette expérience, nous présentons un type de SCWT avec une première tâche stroop
congruente composée de rectangles de différentes couleurs à nommer dans une lecture à
voix haute et une seconde incongruente qui consiste à nommer la couleur de l’encre en
ignorant la réponse automatique provoquée par la lecture automatisée du mot incongruent
en mesurant la variable dépendante à savoir le temps mis par le participant.e.s à accomplir
la tâche. On mesure le temps mis pour les deux tâches puis on regarde si la moyenne des
temps pour la série incongruente est inférieure de plus de 5% par rapport à la congruente.
L’étude se concentre sur une population d’adultes jeunes sans problème visuel ou
attentionnel avec un échantillon de 18 personnes. On supprime les variables parasites de
bruit ambiant et autres distractions en effectuant l’expérience dans une pièce calme, l’effet
d’ordre est supprimé par le fait que certains commencent l’expérience avec les rectangles et
d’autres commencent par les mots. De plus, on ne compte pas les erreurs donc on insiste
sur la nécessité d’être précis plutôt que rapide quitte à se corriger si besoin afin que les
données de temps soient pertinentes et on néglige la différence d’âge, les participants sont
tous jeunes adultes.

![image](https://github.com/estellelm38/stroop_test/assets/144695298/b3f3b911-b88a-4889-b02a-d7531c70fd3c)

L’objectif de cette recherche est d’abord de montrer que le temps moyen mis pour la tâche
Stroop incongruente est supérieur.
hypothèse théorique : La lecture interfère sur la dénomination des couleurs
hypothèse opérationnelle : Le temps mis par le/la participant.e pour nommer les couleurs
sera plus long lors de la série incongruente que lors de la série congruente
hypothèses statistiques : La moyenne des temps pour la série incongruente est inférieure
de plus de 5% par rapport à la congruence.
L'effet Stroop variera de manière proportionnelle en fonction de la congruence des stimuli.
VI : condition expérimentale (2 modalités : congruent ou incongruent, mot ou rectangle)
VD : temps de réalisation de la tâche
La condition “mot” (effet stroop) provoque une augmentation du temps de lecture des
couleurs.

### 2.2 Transformation des données
La variable d'intérêt, appelée "Inhibition," a été calculée comme la différence entre le temps
de réaction incongruent et le temps de réaction congruent pour chaque participant. Ainsi,
Inhibition = Incongruent - Congruent. Cette variable mesure le temps supplémentaire
nécessaire pour traiter des stimuli incongruents par rapport aux stimuli congruents.

### 2.3 Analyse de régression linéaire multiple
L'analyse de régression linéaire multiple a été réalisée pour explorer la relation entre les
variables explicatives (Congruent, CongruentSquared) et la variable dépendante (Inhibition).
Le modèle est formulé comme suit :
Inhibition = X + Y * Congruent + Z * CongruentSquared + ε
● X est l'intercept
● Y est le coefficient de la variable Congruent
● Z est le coefficient de la variable CongruentSquared
● ε est le terme d'erreur.

### 2.4 Validation des hypothèses
Avant d'interpréter les résultats, des vérifications ont été effectuées pour s'assurer que les
hypothèses de la régression linéaire multiple étaient satisfaites. Cela inclut l'examen des
résidus pour évaluer la normalité et l'homoscédasticité, ainsi que la vérification de
l'indépendance des erreurs.
Le programme a été rédigé en langage python version 3.8.18 et exécuté sur le logiciel
conda.
Le programme suivant contient les classes :
- Agent : avec des méthodes de lecture du mot, reconnaissance de couleur et
comparaison des deux ;
- SimpleNet : c’est le réseau de neurones ;
- Training : qui permet d’entraîner le réseau sur un dossier contenant les images ;
