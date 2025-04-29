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

Le script principal est `Basecode/main.py`. Vous pouvez le lancer de deux manières :

1.  **Mode Entraînement (par défaut) :**
    Lance un nouvel entraînement complet de l'IA en utilisant l'algorithme génétique. Les paramètres d'entraînement (nombre de générations, taille de la population, etc.) sont définis dans `Basecode/main.py` et `Basecode/genetic.py`. Une fois l'entraînement terminé, le meilleur réseau de neurones trouvé est sauvegardé dans `model.txt` et la visualisation du modèle jouant au Snake est lancée.

    ```bash
    python Basecode/main.py
    ```

2.  **Mode Chargement et Visualisation :**
    Charge un modèle précédemment entraîné et sauvegardé dans `model.txt` et lance directement la visualisation sans refaire l'entraînement. Utile pour revoir les performances d'un modèle déjà entraîné.

    ```bash
    python Basecode/main.py --load
    ```

    *Note :* Ce mode échouera si le fichier `model.txt` n'existe pas.

## Structure du Code (Basecode/)

*   `main.py`: Point d'entrée principal, gère les arguments, lance l'entraînement ou le chargement, et la boucle de visualisation.
*   `genetic.py`: Contient l'implémentation de l'algorithme génétique (initialisation, évaluation, sélection, croisement, mutation, intensification, réoptimisation).
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