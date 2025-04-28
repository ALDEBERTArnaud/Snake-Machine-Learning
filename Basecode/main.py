from snake import *
from vue import *
import genetic 
from Utils import *
import sys
import multiprocessing # Import nécessaire
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

# Paramètres de l'évaluation : 
# Taille de grille standard 10x10 pour entraînement classique
gameParams={"nbGames":10, "height":10, "width":10}

# --- Début du bloc principal ---
if __name__ == '__main__':
    multiprocessing.freeze_support() # Nécessaire pour le multi-processing sous Windows
    print(f"Configuration d'entraînement: {gameParams}")

    #fonction d'optimisation, renvoie un réseau de neurones entrainé sur le jeu
    # L'appel à optimize est actif pour un entraînement standard.
    nn = genetic.optimize(
        taillePopulation=400, 
        tailleSelection=50, 
        pc=0.8, 
        mr=2.0, 
        arch=[nbFeatures, 24, nbActions], 
        gameParams=gameParams, 
        nbIterations=400, 
        nbThreads=10, 
        scoreMax=1.0
        # load_model_path="model.txt" # Pour réoptimisation
    )
    #sauvegarde du réseau entraîné
    save_nn(nn, "model.txt") 

    # print("Chargement du modèle depuis model.txt...")
    # try:
    #     nn = load_nn("model.txt")
    #     print("Modèle chargé.")
    # except FileNotFoundError:
    #     print("ERREUR: Le fichier model.txt n'a pas été trouvé.")
    #     sys.exit(1) 
    # except Exception as e:
    #     print(f"ERREUR: {e}")
    #     sys.exit(1)


    #Test visuel, on voit le réseau jouer en temps réel
    print("Lancement de la visualisation...")
    vue = SnakeVue(gameParams["height"], gameParams["width"], 64)
    fps = pygame.time.Clock()
    gameSpeed = 20

    while True:
        game = Game(gameParams["height"], gameParams["width"])
        while game.enCours:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    pygame.quit()
                    sys.exit(0)
            
            # Utiliser le réseau chargé (ou entraîné)
            pred = nn.predict(game.getFeatures())
            game.direction = pred
            game.refresh()
            if not game.enCours: break
            vue.displayGame(game)
            fps.tick(gameSpeed)
            # Limite de pas pour la visualisation
            if game.steps > gameParams["height"] * gameParams["width"] * 2: 
                print("Fin de la partie (limite de pas atteinte)") 
                game.enCours = False 
                break
        
        final_score_val = game.score 
        print(f"Partie terminée. Score final: {final_score_val}")
        
        print("Nouvelle partie de visualisation...")
        pygame.time.wait(100) 
# --- Fin du bloc principal ---

