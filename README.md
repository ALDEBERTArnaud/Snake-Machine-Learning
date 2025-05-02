# Projet Snake IA - Algorithme Génétique

Ce projet vise à entraîner une intelligence artificielle (IA) basée sur un réseau de neurones à jouer au jeu Snake en utilisant un algorithme génétique.

## Prérequis

*   Python 3
*   pip (généralement inclus avec Python)
*   Les bibliothèques `numpy` et `pygame`

## Installation des dépendances

Ouvrez un terminal et exécutez la commande suivante pour installer les bibliothèques nécessaires :

```bash
pip install numpy pygame
```

## Exécution du programme

Le script principal est `Basecode/main.py`. Vous pouvez le lancer de trois manières :

1.  **Mode Entraînement Standard (par défaut) :**
    Lance un nouvel entraînement complet de l'IA (grille 10x10 par défaut) en utilisant l'algorithme génétique. Le meilleur réseau trouvé est sauvegardé dans `model.txt` et la visualisation est lancée.

    ```bash
    python Basecode/main.py
    ```

2.  **Mode Chargement et Visualisation (`--load`) :**
    Charge le modèle précédemment sauvegardé dans `model.txt` et lance directement la visualisation sans refaire l'entraînement.

    ```bash
    python Basecode/main.py --load
    ```

    *Note :* Ce mode échouera si le fichier `model.txt` n'existe pas.

3.  **Mode Réoptimisation (`--reoptimize`) :**
    Charge le modèle existant dans `model.txt` et l'utilise comme point de départ pour continuer l'entraînement, potentiellement avec des paramètres différents (par exemple, une taille de grille de 15x15 par défaut dans ce mode). Le modèle réoptimisé est ensuite sauvegardé dans `model.txt` (écrasant l'ancien) et la visualisation est lancée.

    ```bash
    python Basecode/main.py --reoptimize
    ```
    *Note :* Ce mode échouera également si le fichier `model.txt` n'existe pas.


## Structure du Code (Basecode/)

*   `main.py`: Point d'entrée principal, gère les arguments (`--load`, `--reoptimize`), lance l'entraînement ou le chargement, et la boucle de visualisation.
*   `genetic.py`: Contient l'implémentation de l'algorithme génétique (initialisation, évaluation, sélection, croisement, mutation, intensification, réoptimisation à partir d'un modèle chargé).
    *   *Note sur le Croisement :* Le code inclut deux types de croisements (mélange basé sur alpha et échange de couches). Après tests, le mélange basé sur alpha seul semble conduire à une convergence plus rapide et est donc actif par défaut. L'option d'utiliser un mélange des deux est conservée en commentaire dans `genetic.py` à des fins de démonstration.
    *   *Note sur l'Intensification :* Le code implémente également une phase d'intensification (recherche locale autour du meilleur individu). Cependant, les tests initiaux suggèrent qu'elle n'améliore pas significativement la convergence dans la configuration actuelle et peut ralentir le processus global. Elle est donc désactivée par défaut (`intensification_freq=0` dans `main.py`) mais reste disponible pour expérimentation.
*   `snake.py`: Définit la logique du jeu Snake (classe `Game`).
*   `NeuralNetwork.py` / `Dense.py`: Implémentation des réseaux de neurones.
*   `vue.py`: Gère l'affichage graphique du jeu avec Pygame.
*   `Utils.py`: Fonctions utilitaires (ex: sauvegarde/chargement de modèle).
*   `Fonctions.py`: Fonctions d'activation pour les réseaux.
*   `snake.png`: Feuille de sprites pour l'affichage graphique.
*   `model.txt`: Fichier où le meilleur modèle entraîné est sauvegardé (et chargé).

## Visualisation

Pendant la phase de visualisation :
*   Appuyez sur `q` pour quitter.
*   Le score final de chaque partie est affiché dans la console avant le début de la suivante. 