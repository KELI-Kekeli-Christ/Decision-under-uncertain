# Planification de prises de vue SPOT-5 avec incertitude (nuages + pannes)
#
# Ce script utilise la Programmation Linéaire Mixte (MILP) avec pyscipopt.
# L'incertitude probabiliste imprécise des prévisions météorologiques est 
# modélisée via l'intégrale de Choquet-Hurwicz (capacité paramétrée par alpha).
#
# Usage :
#   python spotProba.py spotProba4 --criterion choquet --alpha 1.0   # (Choquet Pessimiste pur / Bel)
#   python spotProba.py spotProba5 --criterion choquet --alpha 0.0   # (Choquet Optimiste pur / Pl)
#   python spotProba.py spotProba5 --criterion choquet --alpha 0.75  # (Aversion modérée à l'ambiguïté)
#   python spotProba.py spotProba1 --criterion det                   # (Modèle sans nuages)
#   python spotProba.py --all

import argparse
import importlib
from itertools import combinations

from pyscipopt import Model, quicksum

# Le critère unifié "choquet" remplace toutes les anciennes déclinaisons d'ignorance.
CRITERIA = ("det", "choquet")
STEREO_INSTRUMENTS = (0, 2)  # Correspond aux instruments 1 (avant) et 3 (arrière)


def get_choquet_capacity_for_clouds(d, criterion, alpha):
    """
    Détermine la probabilité d'échec équivalente pour calculer la capacité
    de succès dans l'intégrale de Choquet-Hurwicz.
    
    Capacité(Succès) = alpha * P_*(Succès) + (1 - alpha) * P^*(Succès)
                     = alpha * (1 - ProbaSup) + (1 - alpha) * (1 - ProbaInf)
                     = 1 - [alpha * ProbaSup + (1 - alpha) * ProbaInf]
    """
    n = d.nbImages
    if criterion == "choquet":
        # p_eq représente le terme entre crochets de la dérivation mathématique
        return [alpha * d.ProbaSup[i] + (1 - alpha) * d.ProbaInf[i] for i in range(n)]
    
    # Cas déterministe (aucune probabilité de nuages considérée)
    return [0.0] * n


def in_conflict(d, i, j, k):
    """
    Vérifie si deux acquisitions i et j sur l'instrument k se chevauchent physiquement.
    L'inéquation est réécrite sans division pour garantir la stabilité numérique.
    """
    return abs(d.DD[i][k] - d.DD[j][k]) * d.VI < d.DU * d.VI + abs(d.AN[i][k] - d.AN[j][k])


def gain_coefficients(d, criterion, alpha):
    """
    Calcule les coefficients de la fonction objectif (Intégrale de Choquet-Hurwicz).
    Retourne deux dictionnaires : un pour les images mono, un pour les stéréo.
    """
    p = get_choquet_capacity_for_clouds(d, criterion, alpha)
    use_failure = criterion != "det"
    f = d.Failure if use_failure else [0.0] * d.nbInstruments
    
    c_mono, c_stereo = {}, {}
    for i in range(d.nbImages):
        if d.TY[i] == 1:
            for k in range(d.nbInstruments):
                # Ch_mu(d) = Gain * Capacité_alpha(Ciel clair) * P(Instrument OK)
                c_mono[i, k] = d.PA[i] * (1 - p[i]) * (1 - f[k])
        else:
            # Pour la stéréo, on multiplie les probabilités de bon fonctionnement 
            # des deux instruments requis (avant et arrière).
            c_stereo[i] = d.PA[i] * (1 - p[i]) * (1 - f[0]) * (1 - f[2])
            
    return c_mono, c_stereo


def build_and_solve(d, criterion="choquet", alpha=1.0, verbose=False, cip=None):
    """
    Construit et résout le modèle MILP.
    """
    n, m = d.nbImages, d.nbInstruments
    model = Model()
    if not verbose:
        model.hideOutput()

    # Variables de décision booléennes
    y = {i: model.addVar(vtype="B", name=f"select{i}") for i in range(n)}
    x = {(i, k): model.addVar(vtype="B", name=f"assignto{i}_{k}")
         for i in range(n) for k in range(m)}

    # Fonction Objectif
    c_mono, c_stereo = gain_coefficients(d, criterion, alpha)
    model.setObjective(
        quicksum(c * x[i, k] for (i, k), c in c_mono.items())
        + quicksum(c * y[i] for i, c in c_stereo.items()),
        sense="maximize",
    )

    # Contraintes d'affectation
    for i in range(n):
        if d.TY[i] == 1:
            # Type Mono : Assignation à exactement 1 instrument si sélectionnée
            model.addCons(quicksum(x[i, k] for k in range(m)) == y[i])
        else:
            # Type Stéréo : Assignation stricte aux instruments 0 et 2
            for k in range(m):
                if k in STEREO_INSTRUMENTS:
                    model.addCons(x[i, k] == y[i])
                else:
                    model.addCons(x[i, k] == 0)

    # Contrainte de mémoire globale
    model.addCons(quicksum(d.PM[i] * y[i] for i in range(n)) <= d.PMmax)

    # Contraintes cinématiques (non-chevauchement des miroirs)
    for k in range(m):
        for i, j in combinations(range(n), 2):
            if (d.TY[i] == 2 or d.TY[j] == 2) and k not in STEREO_INSTRUMENTS:
                continue
            
            if in_conflict(d, i, j, k):
                model.addCons(x[i, k] + x[j, k] <= 1)

    if cip:
        model.writeProblem(cip)
        
    model.optimize()

    status = model.getStatus()
    plan = []
    if status == "optimal":
        for i in range(n):
            for k in range(m):
                if model.getVal(x[i, k]) > 0.5:
                    plan.append((i, k))
                    
    return status, (model.getObjVal() if status == "optimal" else None), plan


def check_plan(d, plan):
    """Vérification indépendante de la faisabilité physique du plan d'acquisition."""
    errors = []
    by_image = {}
    for i, k in plan:
        by_image.setdefault(i, []).append(k)
        
    for i, ks in by_image.items():
        if d.TY[i] == 1 and len(ks) != 1:
            errors.append(f"Erreur Mono : {i} affectée sur {ks}")
        if d.TY[i] == 2 and sorted(ks) != list(STEREO_INSTRUMENTS):
            errors.append(f"Erreur Stéréo : {i} affectée sur {ks}")
            
    if sum(d.PM[i] for i in by_image) > d.PMmax:
        errors.append("Erreur : La capacité mémoire totale est dépassée.")
        
    for k in range(d.nbInstruments):
        imgs = sorted((d.DD[i][k], i) for i, kk in plan if kk == k)
        for (t1, i), (t2, j) in zip(imgs, imgs[1:]):
            need = d.DU + abs(d.AN[i][k] - d.AN[j][k]) / d.VI
            if t2 - t1 < need - 1e-9:
                errors.append(f"Erreur Cinématique (Inst {k}) : collision entre images {i},{j}")
                
    return errors


def expected_gain(d, plan, p):
    """Calcule le gain espéré brut réel d'un plan pour un vecteur de nuages p donné."""
    g = 0.0
    for i, k in plan:
        if d.TY[i] == 1:
            g += d.PA[i] * (1 - p[i]) * (1 - d.Failure[k])
        elif k == STEREO_INSTRUMENTS[0]:
            g += d.PA[i] * (1 - p[i]) * (1 - d.Failure[0]) * (1 - d.Failure[2])
    return g


def report(name, d, criterion, alpha, status, obj, plan):
    label = criterion + (f" (alpha={alpha})" if criterion == "choquet" else "")
    print(f"\n=== {name} | critere {label} | statut {status} ===")
    if status != "optimal":
        return
        
    print(f"Valeur de l'objectif (Intégrale de Choquet unifiée) : {obj:.4f}")
    
    lo = expected_gain(d, plan, d.ProbaSup)
    hi = expected_gain(d, plan, d.ProbaInf)
    print(f"Utilité espérée réelle du plan  : [{lo:.4f}, {hi:.4f}]")
    
    selected = sorted({i for i, _ in plan})
    print(f"Images selectionnées : {len(selected)}/{d.nbImages}, "
          f"Mémoire consommée : {sum(d.PM[i] for i in selected)}/{d.PMmax}")
          
    for k in range(d.nbInstruments):
        seq = sorted((d.DD[i][k], i) for i, kk in plan if kk == k)
        txt = ", ".join(f"img{i}({'S' if d.TY[i] == 2 else 'M'})@{t}" for t, i in seq)
        print(f"  Instrument {k + 1} : {txt if txt else '-'}")
        
    errors = check_plan(d, plan)
    print("Vérification faisabilité :", "OK" if not errors else errors)


def run(name, criterion, alpha, verbose=False, cip=None):
    d = importlib.import_module(name)
    status, obj, plan = build_and_solve(d, criterion, alpha, verbose, cip)
    report(name, d, criterion, alpha, status, obj, plan)
    return obj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("dataset", nargs="?", help="Module de donnees, ex. spotProba4")
    ap.add_argument("--criterion", choices=CRITERIA, default="choquet")
    ap.add_argument("--alpha", type=float, default=1.0, help="Indice de pessimisme (1=Pessimiste pur, 0=Optimiste pur)")
    ap.add_argument("--all", action="store_true", help="Execute tous les jeux et tous les criteres")
    ap.add_argument("--verbose", action="store_true", help="Affiche la sortie du solveur")
    ap.add_argument("--cip", help="Exporte le modele lineaire au format .cip")
    a = ap.parse_args()

    if a.all:
        for name in ("spotProba1", "spotProba2", "spotProba3", "spotProba4", "spotProba5"):
            for crit in CRITERIA:
                run(name, crit, a.alpha)
    elif a.dataset:
        run(a.dataset, a.criterion, a.alpha, a.verbose, a.cip)
    else:
        ap.error("Specifiez un jeu de donnees ou utilisez --all")


if __name__ == "__main__":
    main()