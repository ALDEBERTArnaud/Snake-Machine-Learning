import random
import itertools
import numpy
from NeuralNetwork import *

nbFeatures = 8
nbActions = 4

class Game:
    def __init__(self, hauteur, largeur):
        self.grille = [[0]*hauteur  for _ in range(largeur)]
        self.hauteur, self.largeur = hauteur, largeur
        self.serpent = [[largeur//2-i-1, hauteur//2] for i in range(4)]
        for (x,y) in self.serpent: self.grille[x][y] = 1
        self.direction = 3
        self.accessibles = [[x,y] for (x,y) in list(itertools.product(range(largeur), range(hauteur))) if [x,y] not in self.serpent]
        self.fruit = [0,0]
        self.setFruit()
        self.enCours = True
        self.steps = 0
        self.score = 4
    
    def setFruit(self):
        if (len(self.accessibles)==0): return
        self.fruit = self.accessibles[random.randint(0, len(self.accessibles)-1)][:]
        self.grille[self.fruit[0]][self.fruit[1]] = 2

    def refresh(self):
        nextStep = self.serpent[0][:]
        match self.direction:
            case 0: nextStep[1]-=1
            case 1: nextStep[1]+=1
            case 2: nextStep[0]-=1
            case 3: nextStep[0]+=1

        if nextStep not in self.accessibles:
            self.enCours = False
            return
        self.accessibles.remove(nextStep)
        if self.grille[nextStep[0]][nextStep[1]]==2:
            self.setFruit()
            self.steps = 0
            self.score+=1
        else:
            self.steps+=1
            self.grille[self.serpent[-1][0]][self.serpent[-1][1]] = 0
            self.accessibles.append(self.serpent[-1][:])
            self.serpent = self.serpent[:-1]
            if self.steps>self.hauteur*self.largeur:
                self.enCours = False
                return

        self.grille[nextStep[0]][nextStep[1]] = 1
        self.serpent = [nextStep]+self.serpent

    def getFeatures(self):
        features = numpy.zeros(8, dtype=float)
        tete_x, tete_y = self.serpent[0]

        pos_haut = (tete_x, tete_y - 1)
        features[0] = 1 if (
            pos_haut[1] < 0 or 
            self.grille[pos_haut[0]][pos_haut[1]] == 1 
        ) else 0

        pos_bas = (tete_x, tete_y + 1)
        features[1] = 1 if (
            pos_bas[1] >= self.hauteur or 
            self.grille[pos_bas[0]][pos_bas[1]] == 1
        ) else 0

        pos_gauche = (tete_x - 1, tete_y)
        features[2] = 1 if (
            pos_gauche[0] < 0 or 
            self.grille[pos_gauche[0]][pos_gauche[1]] == 1
        ) else 0

        pos_droite = (tete_x + 1, tete_y)
        features[3] = 1 if (
            pos_droite[0] >= self.largeur or 
            self.grille[pos_droite[0]][pos_droite[1]] == 1
        ) else 0

        fruit_x, fruit_y = self.fruit

        if fruit_y < tete_y:
            features[4] = 1.0 
        elif fruit_y > tete_y:
            features[4] = -1.0 
        else:
            features[4] = 0.0

        if fruit_x > tete_x:
            features[5] = 1.0
        elif fruit_x < tete_x:
            features[5] = -1.0
        else:
            features[5] = 0.0

        features[6] = float(self.direction)

        distance = 0.0
        if self.direction == 0:
            distance = tete_y
            features[7] = distance / self.hauteur
        elif self.direction == 1:
            distance = self.hauteur - 1 - tete_y
            features[7] = distance / self.hauteur
        elif self.direction == 2:
            distance = tete_x
            features[7] = distance / self.largeur
        elif self.direction == 3:
            distance = self.largeur - 1 - tete_x
            features[7] = distance / self.largeur

        return features
    
    def print(self):
        print("".join(["="]*(self.largeur+2)))
        for ligne in range(self.hauteur):
            chaine = ["="]
            for colonne in range(self.largeur):
                if self.grille[colonne][ligne]==1: chaine.append("#")
                elif self.grille[colonne][ligne]==2: chaine.append("F")
                else: chaine.append(" ")
            chaine.append("=")
            print("".join(chaine))
        print("".join(["="]*(self.largeur+2))+"\n")

