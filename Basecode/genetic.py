import numpy
from NeuralNetwork import *
from snake import *
import random
import copy
from concurrent.futures import ProcessPoolExecutor
import os
from Utils import load_nn

def evaluate_individual_helper(args):
    individu, game_params = args
    return eval(individu, game_params)

def eval(sol, gameParams):
    nbGames = gameParams['nbGames']
    height = gameParams['height']
    width = gameParams['width']
    nn = sol.nn

    total_score_sum = 0
    initial_score = 4

    for _ in range(nbGames):
        game = Game(height, width)
        while game.enCours:
            features = game.getFeatures()
            prediction_index = nn.predict(features)
            if isinstance(prediction_index, numpy.ndarray):
                 game.direction = numpy.argmax(prediction_index)
            else:
                 game.direction = prediction_index

            game.refresh()

        pi = game.score - initial_score
        si = game.steps

        total_score_sum += (1000 * pi + si)

    score = total_score_sum / (nbGames * height * width * 1000)
    return score

'''
Représente une solution avec
_un réseau de neurones
_un score (à maximiser)

vous pouvez ajouter des attributs ou méthodes si besoin
'''
class Individu:
    def __init__(self, nn):
        self.nn = nn
        self.score = 0


'''
La méthode d'initialisation de la population est donnée :
_on génère N individus contenant chacun un réseau de neurones (de même format)
_on évalue et on trie des individus
'''
def initialization(taillePopulation, arch, gameParams, nbThreads, base_nn=None, initial_mutation_mr=0.5):
    population = []
    for i in range(taillePopulation):
        if base_nn is not None:
            nn = copy.deepcopy(base_nn)
            temp_individu = Individu(nn)
            mutation(temp_individu, initial_mutation_mr, arch)
            population.append(temp_individu)
        else:
            nn = NeuralNetwork((arch[0],))
            for j in range(1, len(arch)):
                nn.addLayer(arch[j], "elu")
            population.append(Individu(nn))

    scores = []
    args_list = [(ind, gameParams) for ind in population]
    effective_threads = min(nbThreads, os.cpu_count() if os.cpu_count() else 1) 
    with ProcessPoolExecutor(max_workers=effective_threads) as executor:
        scores = list(executor.map(evaluate_individual_helper, args_list))

    for i, score in enumerate(scores):
        population[i].score = score

    population.sort(reverse=True, key=lambda sol:sol.score)
    
    return population

def mutation(individu, mr, arch):
    nn = individu.nn
    mutation_scale = 0.1

    for l in range(len(nn.layers)):
        layer = nn.layers[l]
        prevLayerSize = arch[l]
        layerSize = arch[l+1]

        if layerSize > 0:
             pm_biais = mr / layerSize
             mask_biais = numpy.random.random(layer.bias.shape) < pm_biais
             random_change_biais = numpy.random.randn(*layer.bias.shape) * mutation_scale
             layer.bias += mask_biais * random_change_biais

        if prevLayerSize > 0:
             pm_poids = mr / prevLayerSize
             mask_poids = numpy.random.random(layer.weights.shape) < pm_poids
             random_change_poids = numpy.random.randn(*layer.weights.shape) * mutation_scale
             layer.weights += mask_poids * random_change_poids

def crossover_layer_swap(parent1_nn, parent2_nn):
    enfant1_nn = copy.deepcopy(parent1_nn)
    enfant2_nn = copy.deepcopy(parent2_nn)

    for l in range(len(parent1_nn.layers)):
        if l % 2 == 0:
            enfant2_nn.layers[l] = copy.deepcopy(parent2_nn.layers[l])
        else:
            enfant1_nn.layers[l] = copy.deepcopy(parent2_nn.layers[l])
            
    return enfant1_nn, enfant2_nn

def crossover_alpha(parent1_nn, parent2_nn):
    enfant1_nn = copy.deepcopy(parent1_nn)
    enfant2_nn = copy.deepcopy(parent2_nn)

    for l in range(len(parent1_nn.layers)):
        layer_p1 = parent1_nn.layers[l]
        layer_p2 = parent2_nn.layers[l]
        layer_c1 = enfant1_nn.layers[l]
        layer_c2 = enfant2_nn.layers[l]

        alpha = random.random()

        w_p1 = layer_p1.weights
        w_p2 = layer_p2.weights
        layer_c1.weights = alpha * w_p1 + (1 - alpha) * w_p2
        layer_c2.weights = (1 - alpha) * w_p1 + alpha * w_p2

        b_p1 = layer_p1.bias
        b_p2 = layer_p2.bias
        layer_c1.bias = alpha * b_p1 + (1 - alpha) * b_p2
        layer_c2.bias = (1 - alpha) * b_p1 + alpha * b_p2
        
    return enfant1_nn, enfant2_nn

def optimize(taillePopulation, tailleSelection, pc, mr, arch, gameParams, nbIterations, nbThreads, scoreMax, intensification_freq=50, intensification_iter=10, intensification_clones=100, load_model_path=None):
    effective_threads = min(nbThreads, os.cpu_count() if os.cpu_count() else 1) 
    print(f"Using {effective_threads} parallel processes for evaluation.")

    base_model = None
    if load_model_path is not None:
        try:
            print(f"Attempting to load base model from: {load_model_path}")
            base_model = load_nn(load_model_path)
            print("Base model loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load base model from {load_model_path}. Starting from scratch. Error: {e}")
            base_model = None

    population = initialization(taillePopulation, arch, gameParams, effective_threads, base_nn=base_model)

    initial_score_display = population[0].score if population else 'N/A'
    print(f"Generation 0 (Initial Population), Best Score: {initial_score_display}")
    
    for generation in range(nbIterations):
        meilleurs = population[:tailleSelection]

        enfants = []
        while len(enfants) < taillePopulation - tailleSelection:
            idx1 = random.randrange(tailleSelection)
            idx2 = random.randrange(tailleSelection)
            parent1 = meilleurs[idx1]
            parent2 = meilleurs[idx2]

            enfant1_nn = copy.deepcopy(parent1.nn)
            enfant2_nn = copy.deepcopy(parent2.nn)

            if random.random() < pc:
                # Par défaut, seul crossover_alpha est actif car il semble plus performant.
                # Pour tester le mix, décommentez les lignes ci-dessous.
                # if random.random() < 0.5: 
                enfant1_nn, enfant2_nn = crossover_alpha(parent1.nn, parent2.nn)
                # else:
                #     enfant1_nn, enfant2_nn = crossover_layer_swap(parent1.nn, parent2.nn)

            enfant1 = Individu(enfant1_nn)
            enfant2 = Individu(enfant2_nn)

            mutation(enfant1, mr, arch)
            mutation(enfant2, mr, arch)

            enfants.append(enfant1)
            if len(enfants) < taillePopulation - tailleSelection:
                enfants.append(enfant2)

        scores_enfants = []
        args_list_enfants = [(enfant, gameParams) for enfant in enfants]
        with ProcessPoolExecutor(max_workers=effective_threads) as executor:
            scores_enfants = list(executor.map(evaluate_individual_helper, args_list_enfants))
        
        for i, score in enumerate(scores_enfants):
            enfants[i].score = score

        population = meilleurs + enfants
        population.sort(reverse=True, key=lambda sol:sol.score)
        current_best_score = population[0].score

        # --- Phase d'Intensification (périodique, désactivée par défaut via intensification_freq=0 dans main.py) ---
        if intensification_freq > 0 and (generation + 1) % intensification_freq == 0:
            print(f"--- Starting Intensification Phase (Generation {generation+1}) ---")
            point_reference = copy.deepcopy(population[0])
            initial_intensification_score = point_reference.score

            for i_iter in range(intensification_iter):
                clones_intensification = []
                for _ in range(intensification_clones):
                    clone = Individu(copy.deepcopy(point_reference.nn))
                    mutation(clone, mr, arch)
                    clones_intensification.append(clone)
                
                scores_clones = []
                args_list_clones = [(clone, gameParams) for clone in clones_intensification]
                with ProcessPoolExecutor(max_workers=effective_threads) as executor:
                     scores_clones = list(executor.map(evaluate_individual_helper, args_list_clones))
                
                best_clone_score = -1
                best_clone_index = -1
                for i, score in enumerate(scores_clones):
                    clones_intensification[i].score = score
                    if score > best_clone_score:
                        best_clone_score = score
                        best_clone_index = i

                if best_clone_index != -1 and best_clone_score > point_reference.score:
                    point_reference = copy.deepcopy(clones_intensification[best_clone_index])
                    print(f"  Intensification Iter {i_iter+1}/{intensification_iter}: New best score {point_reference.score:.6f}")
                else:
                    print(f"  Intensification Iter {i_iter+1}/{intensification_iter}: Score unchanged ({point_reference.score:.6f})")
            
            if point_reference.score > current_best_score:
                print(f"--- Intensification improved score from {current_best_score:.6f} to {point_reference.score:.6f} ---")
                population[0] = point_reference
            else:
                 print(f"--- Intensification finished, no improvement found (Best: {current_best_score:.6f}) ---")

        print(f"Generation {generation+1}/{nbIterations}, Best Score: {population[0].score:.6f}")

        if population[0].score >= scoreMax:
            print(f"Score maximum {scoreMax} atteint à la génération {generation+1}")
            break

    return population[0].nn
