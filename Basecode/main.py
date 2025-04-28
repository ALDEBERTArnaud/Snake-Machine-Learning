from snake import *
from vue import *
import genetic 
from Utils import *
import sys
import multiprocessing
import os
import argparse
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

# Paramètres de l'évaluation : 
# Taille de grille standard 10x10 pour entraînement classique
gameParams={"nbGames":10, "height":10, "width":10}

# --- Début du bloc principal ---
if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    # --- Configuration des arguments de ligne de commande ---
    parser = argparse.ArgumentParser(description='Entraînement ou visualisation d\'un Snake IA.')
    parser.add_argument('--load', action='store_true',
                        help='Charge le modèle existant (model.txt) et visualise sans entraîner.')
    args = parser.parse_args()
    # ------------------------------------------------------

    print(f"Configuration d'entraînement/jeu: {gameParams}") 

    nn = None # Initialiser nn

    if args.load:
        # --- Mode Chargement --- 
        print("Mode Chargement sélectionné.")
        print("Chargement du modèle depuis model.txt...")
        try:
            nn = load_nn("model.txt")
            print("Modèle chargé.")
        except FileNotFoundError:
            print("ERREUR: Le fichier model.txt n'a pas été trouvé. Impossible de charger.")
            sys.exit(1) 
        except Exception as e:
            print(f"ERREUR: Une erreur est survenue lors du chargement du modèle : {e}")
            sys.exit(1)
    else:
        # --- Mode Entraînement (par défaut) --- 
        print("Mode Entraînement sélectionné.")
        #fonction d'optimisation, renvoie un réseau de neurones entrainé sur le jeu
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
        print("Sauvegarde du modèle entraîné dans model.txt...")
        save_nn(nn, "model.txt") 
        print("Modèle sauvegardé.")

    # --- Test visuel --- 
    if nn is None:
        print("Erreur: Aucun réseau neuronal n'a été chargé ou entraîné.")
        sys.exit(1)
        
    print("Lancement de la visualisation...")
    vue = SnakeVue(gameParams["height"], gameParams["width"], 64)
    fps = pygame.time.Clock()
    gameSpeed = 20

    while True:
        # Utiliser les gameParams définis au début (qui pourraient être modifiés pour la visu)
        game = Game(gameParams["height"], gameParams["width"])
        while game.enCours:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    pygame.quit()
                    sys.exit(0)
            
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

