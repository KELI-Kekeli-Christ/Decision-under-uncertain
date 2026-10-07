# Resolution d'un problème de planification de prise de vue  sans incertitude
# Helene Fargier, oct 2025
#
#  Version bootstrap, avec exemple de declaration de modele, ajout de variables de decision, 
# d'une fonction objectif  ne prenant pas en compte les incertitudes 
# une seule contrainte implementée



# on charge le solveur lineaire
from pyscipopt import Model, quicksum
from itertools import product

# on charge les données
from spotProba5 import nbImages, nbInstruments, PA, DD, AN, VI, DU, TY, PM, PMmax, Failure, ProbaInf, ProbaSup


# creation du modele lineaire
#############################

#model
mymodel = Model()

# pour chaque image i ,  le solveur doit affecter la variable booleen selection[i) à 1 ssi  l'image i est selectionnée
selection = {}
for i in range(nbImages):
    selection[i] = mymodel.addVar(vtype='B', name='select' + str(i))

#### pour chaque image i ,  le solveur doit affecter la variable booleen selection[i) à 1 ssi  l'image i est selectionnée
assignedTo = {}
for i in range(nbImages):
    ass_i = {}
    for j in range(nbInstruments):
        ass_i[j] = mymodel.addVar(vtype='B', name='assignto' + str(i) + '_' + str(j))
    assignedTo[i] = ass_i


# la fonction objectif
######################

# en l'absence d'incertitude, on maximise la somme des payoff
# objectif d'amorge A CHANGER ?

# si mono prix d'image * proba de bon temps * proba de non panne ( 1 - panne) de l'instrument au quel on affecte 
# si streo prix * proba bonne temps * proba de non panne de 0 et 2 
# mono c'est TY[I] = 1 , streo TY[I] = 2

alpha = 1 # pessimiste 
proba_bon_temps = [1- (alpha * ProbaSup[i] + (1 - alpha) * ProbaInf[i]) for i in range(nbImages)]
proba_non_panne = [1 - Failure[i] for i in range(nbInstruments)]

# proba bon temps = 1 - (alpha * proba sup + (1-alpha) * proba inf)
mymodel.setObjective(
    quicksum(PA[i] * proba_bon_temps[i] * proba_non_panne[j] * assignedTo[i][j]
    for i in range(nbImages) if TY[i] == 1
    for j in range(nbInstruments))
    + quicksum(PA[i] * proba_bon_temps[i] * proba_non_panne[0] * proba_non_panne[2] * selection[i]
    for i in range(nbImages) if TY[i] == 2)
    , sense='maximize')


# ajout des contraintes au modele
################################

# la contrainte de non chevauchement
# considérons un instrument
# si, sur cet insrument, le temps de transition entre 2 images ima1 et ima2 
# ne tient pas entre la fin de ima1 et le debut de ima2 
# alors une seule de ces deux images au plus peut etre assignée à l'instrument


# Contrainte 1 de non chevechement : pour chaque combinaision d'image  
# ajout de ce que cause probleme au solver comme contraintes 
for ima1,ima2 in product(range(nbImages), range(nbImages)):
    if ima1 < ima2:
        for ins in range(nbInstruments):
            if  abs(DD[ima1][ins] - DD[ima2][ins]) * VI < DU * VI + abs(AN[ima1][ins] - AN[ima2][ins]):
                mymodel.addCons(assignedTo[ima1][ins] + assignedTo[ima2][ins] <= 1)
                

# AJOUTER LES DEUX CONTRAINTE MEMOIRE ET AFFECTATION !!!! 

# Contraint de memoire  
mymodel.addCons(quicksum(PM[i] * selection[i] for i in range(nbImages)) <= PMmax)

#contrainte d'affection : si stereo les images peuvent pas etre affecte au nadir
mymodel.addCons(quicksum(assignedTo[i][1] for i in range(nbImages) if TY[i] == 2) == 0)


# si choisi on doit affecter l'image à un instrument ou deux si stereo
for i in range(nbImages):
    mymodel.addCons(quicksum(assignedTo[i][j] for j in range(nbInstruments)) == TY[i]*selection[i])
                
# resolution et affichage des resulats
#########################################

#visualiser le problem lineaire cree
mymodel.writeProblem("pb.cip")

# lancer l'optimisation
print("Resolution")
mymodel.hideOutput(False)
mymodel.optimize()

#afficiher  les resultats mode "scip"
print('statut ' + mymodel.getStatus())
print("solution", end='\t')
print(mymodel.getBestSol())

# afficher les resultats prorement
if mymodel.getStatus() == 'optimal':
    print("\n\nProblème resolu, valeur de l'objectif " + str(mymodel.getObjVal()))
    sol=mymodel.getBestSol()
    for ima in range(nbImages):
        for ins in range(nbInstruments):
            if (mymodel.getVal(assignedTo[ima][ins]) > 0):
                print("Image" + str(ima) + " selectionnée et  assignée à  " + str(ins) + "  (debut à " + str( DD[ima][ins]) + ")")

