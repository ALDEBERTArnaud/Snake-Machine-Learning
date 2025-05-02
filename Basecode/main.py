from snake import *
from vue import *
import genetic 
from Utils import *
import sys
import multiprocessing
import os
import argparse
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

DEFAULT_GAME_PARAMS = {"nbGames":10, "height":10, "width":10}
DEFAULT_ITERATIONS = 500

REOPTIMIZE_GAME_PARAMS = {"nbGames":10, "height":15, "width":15}
REOPTIMIZE_ITERATIONS = 300

MODEL_FILENAME = "model.txt"

if __name__ == '__main__':
    multiprocessing.freeze_support() 
    
    parser = argparse.ArgumentParser(description='Entraînement ou visualisation d\'un Snake IA.')
    parser.add_argument('--load', action='store_true',
                        help=f'Charge le modèle existant ({MODEL_FILENAME}) et visualise sans entraîner.')
    parser.add_argument('--reoptimize', action='store_true',
                        help=f'Charge le modèle existant ({MODEL_FILENAME}) et continue l\'entraînement (réoptimisation), potentiellement avec des paramètres différents.')
    args = parser.parse_args()

    nn = None 
    current_game_params = None 
    mode = ""

    if args.load:
        mode = "Chargement & Visualisation"
        current_game_params = DEFAULT_GAME_PARAMS 
        print(f"Mode sélectionné: {mode}")
        print(f"Affichage avec config: {current_game_params}")
        print(f"Chargement du modèle depuis {MODEL_FILENAME}...")
        try:
            nn = load_nn(MODEL_FILENAME)
            print("Modèle chargé.")
        except FileNotFoundError:
            print(f"ERREUR: Le fichier {MODEL_FILENAME} n'a pas été trouvé. Impossible de charger.")
            sys.exit(1) 
        except Exception as e:
            print(f"ERREUR: Une erreur est survenue lors du chargement du modèle : {e}")
            sys.exit(1)
    
    elif args.reoptimize:
        mode = "Réoptimisation"
        current_game_params = REOPTIMIZE_GAME_PARAMS 
        print(f"Mode sélectionné: {mode}")
        print(f"Configuration pour réoptimisation: {current_game_params}")
        print(f"Tentative de chargement de {MODEL_FILENAME} comme base...")
        nn = genetic.optimize(
            taillePopulation=400, 
            tailleSelection=50, 
            pc=0.8, 
            mr=2.0, 
            arch=[nbFeatures, 24, nbActions], 
            gameParams=current_game_params, 
            nbIterations=REOPTIMIZE_ITERATIONS, 
            nbThreads=10, 
            scoreMax=1.0,
            load_model_path=MODEL_FILENAME 
        )
        print(f"Sauvegarde du modèle réoptimisé dans {MODEL_FILENAME}...")
        save_nn(nn, MODEL_FILENAME) 
        print("Modèle sauvegardé.")

    else:
        mode = "Entraînement Standard"
        current_game_params = DEFAULT_GAME_PARAMS 
        print(f"Mode sélectionné: {mode}")
        print(f"Configuration pour entraînement: {current_game_params}")
        nn = genetic.optimize(
            taillePopulation=400, 
            tailleSelection=50, 
            pc=0.8, 
            mr=2.0, 
            arch=[nbFeatures, 24, nbActions], 
            gameParams=current_game_params, 
            nbIterations=DEFAULT_ITERATIONS, 
            nbThreads=10, 
            scoreMax=1.0,
            intensification_freq=0
        )
        print(f"Sauvegarde du modèle entraîné dans {MODEL_FILENAME}...")
        save_nn(nn, MODEL_FILENAME) 
        print("Modèle sauvegardé.")

    if nn is None:
        print("Erreur: Aucun réseau neuronal n'a été chargé ou entraîné.")
        sys.exit(1)
        
    print(f"Lancement de la visualisation (Mode: {mode})...")
    vue = SnakeVue(current_game_params["height"], current_game_params["width"], 64)
    fps = pygame.time.Clock()
    gameSpeed = 20

    while True:
        game = Game(current_game_params["height"], current_game_params["width"])
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
            if game.steps > current_game_params["height"] * current_game_params["width"] * 2: 
                print("Fin de la partie (limite de pas atteinte)") 
                game.enCours = False 
                break
        
        final_score_val = game.score 
        print(f"Partie terminée. Score final: {final_score_val}")
        
        print("Nouvelle partie de visualisation...")
        pygame.time.wait(100) 

