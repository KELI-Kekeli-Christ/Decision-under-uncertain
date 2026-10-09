# Planification des acquisitions SPOT-5 en tenant compte des nuages et des pannes
#
# Le problème est formulé comme un programme linéaire en nombres entiers
# mixtes (MILP) et résolu avec PySCIPOpt.
#
# Deux critères sont disponibles :
# - det : planification déterministe, sans prise en compte des nuages ni des pannes.
# - choquet : prise en compte des incertitudes météorologiques à l'aide
#   de l'intégrale de Choquet-Hurwicz, avec un paramètre alpha qui règle
#   le compromis entre une approche pessimiste et une approche optimiste.
#
# Exemples d'utilisation :
#   python spotProba.py spotProba4 --criterion choquet --alpha 1.0
#   python spotProba.py spotProba5 --criterion choquet --alpha 0.0
#   python spotProba.py spotProba5 --criterion choquet --alpha 0.75
#   python spotProba.py spotProba1 --criterion det
#   python spotProba.py --all

import argparse
import importlib
from itertools import combinations

from pyscipopt import Model, quicksum

CRITERIA = ("det", "choquet")

# Une acquisition stéréo nécessite les instruments avant et arrière,
# identifiés ici par les indices 0 et 2.
STEREO_INSTRUMENTS = (0, 2)


def get_choquet_capacity_for_clouds(d, criterion, alpha):
    """
    Calcule le terme lié à l'incertitude météorologique utilisé dans
    la fonction objectif.

    Pour le critère de Choquet-Hurwicz, alpha permet de combiner les
    bornes inférieure et supérieure des probabilités de nuages.

    Le résultat correspond à la probabilité de rencontrer des nuages
    retenue dans le modèle. La capacité associée à un ciel dégagé est
    donc égale à 1 moins cette valeur.

    Avec le critère déterministe, les nuages ne sont pas pris en compte.
    """
    n = d.nbImages

    if criterion == "choquet":
        return [
            alpha * d.ProbaSup[i] + (1 - alpha) * d.ProbaInf[i]
            for i in range(n)
        ]

    return [0.0] * n


def in_conflict(d, i, j, k):
    """
    Détermine si deux acquisitions i et j se chevauchent sur l'instrument k.

    La condition tient compte de l'écart entre les dates d'acquisition,
    du temps nécessaire au mouvement de l'instrument et de la différence
    d'angle entre les deux prises de vue.

    La formule est écrite sans division afin d'éviter les divisions
    inutiles lors de la vérification des conflits.
    """
    return (
        abs(d.DD[i][k] - d.DD[j][k]) * d.VI
        < d.DU * d.VI + abs(d.AN[i][k] - d.AN[j][k])
    )


def gain_coefficients(d, criterion, alpha):
    """
    Calcule les coefficients de la fonction objectif pour chaque acquisition.

    Pour une image mono, le gain dépend de sa valeur, de la probabilité
    d'obtenir un ciel dégagé et de la disponibilité de l'instrument choisi.

    Une image stéréo nécessite deux instruments. Son gain tient donc
    compte de la disponibilité simultanée des instruments avant et arrière.

    La fonction renvoie deux dictionnaires : l'un pour les acquisitions
    mono et l'autre pour les acquisitions stéréo.
    """
    p = get_choquet_capacity_for_clouds(d, criterion, alpha)

    # Les probabilités de panne sont prises en compte uniquement avec
    # le critère de Choquet.
    use_failure = criterion != "det"
    f = d.Failure if use_failure else [0.0] * d.nbInstruments

    c_mono, c_stereo = {}, {}

    for i in range(d.nbImages):
        if d.TY[i] == 1:
            # Une image mono peut être affectée à n'importe quel instrument.
            for k in range(d.nbInstruments):
                c_mono[i, k] = (
                    d.PA[i]
                    * (1 - p[i])
                    * (1 - f[k])
                )
        else:
            # Une image stéréo doit être acquise avec les deux instruments
            # requis. Le gain tient compte de leurs probabilités de panne.
            c_stereo[i] = (
                d.PA[i]
                * (1 - p[i])
                * (1 - f[0])
                * (1 - f[2])
            )

    return c_mono, c_stereo


def build_and_solve(d, criterion="choquet", alpha=1.0,
                    verbose=False, cip=None):
    """
    Construit le modèle de planification, puis le résout avec SCIP.

    Les variables indiquent quelles images sont sélectionnées et quels
    instruments sont utilisés pour les acquérir.

    Le modèle maximise le gain total attendu tout en respectant la capacité
    mémoire, les contraintes d'affectation et les contraintes cinématiques.
    """
    n, m = d.nbImages, d.nbInstruments

    model = Model()

    if not verbose:
        model.hideOutput()

    # y[i] vaut 1 si l'image i est sélectionnée.
    y = {
        i: model.addVar(vtype="B", name=f"select{i}")
        for i in range(n)
    }

    # x[i, k] vaut 1 si l'image i est affectée à l'instrument k.
    x = {
        (i, k): model.addVar(vtype="B", name=f"assignto{i}_{k}")
        for i in range(n)
        for k in range(m)
    }

    # Fonction objectif : maximiser le gain total des acquisitions.
    c_mono, c_stereo = gain_coefficients(d, criterion, alpha)

    model.setObjective(
        quicksum(c * x[i, k] for (i, k), c in c_mono.items())
        + quicksum(c * y[i] for i, c in c_stereo.items()),
        sense="maximize",
    )

    # Contraintes d'affectation des instruments.
    for i in range(n):
        if d.TY[i] == 1:
            # Une image mono sélectionnée doit être affectée à un seul
            # instrument. Si elle n'est pas sélectionnée, aucune affectation
            # n'est possible.
            model.addCons(
                quicksum(x[i, k] for k in range(m)) == y[i]
            )
        else:
            # Une image stéréo doit utiliser les instruments avant et arrière.
            # Elle ne peut pas être affectée aux autres instruments.
            for k in range(m):
                if k in STEREO_INSTRUMENTS:
                    model.addCons(x[i, k] == y[i])
                else:
                    model.addCons(x[i, k] == 0)

    # La mémoire nécessaire à l'ensemble des images sélectionnées
    # ne doit pas dépasser la capacité disponible.
    model.addCons(
        quicksum(d.PM[i] * y[i] for i in range(n)) <= d.PMmax
    )

    # Contraintes cinématiques : deux acquisitions incompatibles ne peuvent
    # pas être programmées sur le même instrument.
    for k in range(m):
        for i, j in combinations(range(n), 2):

            # Une image stéréo n'utilise que les instruments 0 et 2.
            # Il est donc inutile de vérifier les conflits sur les autres.
            if (
                (d.TY[i] == 2 or d.TY[j] == 2)
                and k not in STEREO_INSTRUMENTS
            ):
                continue

            if in_conflict(d, i, j, k):
                model.addCons(x[i, k] + x[j, k] <= 1)

    # Export du modèle pour pouvoir examiner sa formulation si nécessaire.
    if cip:
        model.writeProblem(cip)

    model.optimize()

    status = model.getStatus()
    plan = []

    # On récupère les affectations uniquement si SCIP a trouvé une solution
    # optimale.
    if status == "optimal":
        for i in range(n):
            for k in range(m):
                if model.getVal(x[i, k]) > 0.5:
                    plan.append((i, k))

    objective = model.getObjVal() if status == "optimal" else None

    return status, objective, plan


def check_plan(d, plan):
    """
    Vérifie que le plan obtenu respecte les principales contraintes physiques.

    Cette vérification est effectuée indépendamment du solveur afin de
    détecter d'éventuelles erreurs dans les affectations, la mémoire ou
    l'enchaînement des acquisitions.
    """
    errors = []
    by_image = {}

    # Regroupe les instruments affectés à chaque image.
    for i, k in plan:
        by_image.setdefault(i, []).append(k)

    # Vérification du nombre d'instruments utilisés pour chaque image.
    for i, ks in by_image.items():
        if d.TY[i] == 1 and len(ks) != 1:
            errors.append(f"Erreur Mono : {i} affectée sur {ks}")

        if (
            d.TY[i] == 2
            and sorted(ks) != list(STEREO_INSTRUMENTS)
        ):
            errors.append(f"Erreur Stéréo : {i} affectée sur {ks}")

    # Vérification de la capacité mémoire.
    if sum(d.PM[i] for i in by_image) > d.PMmax:
        errors.append("Erreur : La capacité mémoire totale est dépassée.")

    # Vérification des temps de séparation entre acquisitions successives.
    for k in range(d.nbInstruments):
        imgs = sorted(
            (d.DD[i][k], i)
            for i, kk in plan
            if kk == k
        )

        for (t1, i), (t2, j) in zip(imgs, imgs[1:]):
            need = (
                d.DU
                + abs(d.AN[i][k] - d.AN[j][k]) / d.VI
            )

            if t2 - t1 < need - 1e-9:
                errors.append(
                    f"Erreur Cinématique (Inst {k}) : "
                    f"collision entre images {i},{j}"
                )

    return errors


def expected_gain(d, plan, p):
    """
    Calcule le gain attendu du plan pour un ensemble donné de probabilités
    de nuages.

    Cette fonction permet notamment d'évaluer le même plan avec les bornes
    supérieure et inférieure des probabilités, indépendamment du critère
    utilisé lors de l'optimisation.
    """
    g = 0.0

    for i, k in plan:
        if d.TY[i] == 1:
            # Pour une image mono, seul l'instrument affecté intervient.
            g += (
                d.PA[i]
                * (1 - p[i])
                * (1 - d.Failure[k])
            )

        elif k == STEREO_INSTRUMENTS[0]:
            # Pour une image stéréo, le gain est compté une seule fois,
            # même si deux instruments sont utilisés.
            g += (
                d.PA[i]
                * (1 - p[i])
                * (1 - d.Failure[0])
                * (1 - d.Failure[2])
            )

    return g


def report(name, d, criterion, alpha, status, obj, plan):
    """
    Affiche les résultats de l'optimisation : gain obtenu, images retenues,
    mémoire utilisée, affectation des instruments et faisabilité du plan.
    """
    label = (
        criterion + f" (alpha={alpha})"
        if criterion == "choquet"
        else criterion
    )

    print(f"\n=== {name} | critere {label} | statut {status} ===")

    if status != "optimal":
        return

    print(f"Valeur de l'objectif (Intégrale de Choquet unifiée) : {obj:.4f}")

    # Évalue le plan avec les deux bornes des probabilités de nuages.
    lo = expected_gain(d, plan, d.ProbaSup)
    hi = expected_gain(d, plan, d.ProbaInf)

    print(f"Utilité espérée réelle du plan  : [{lo:.4f}, {hi:.4f}]")

    selected = sorted({i for i, _ in plan})

    print(
        f"Images selectionnées : {len(selected)}/{d.nbImages}, "
        f"Mémoire consommée : "
        f"{sum(d.PM[i] for i in selected)}/{d.PMmax}"
    )

    # Affiche les acquisitions dans l'ordre chronologique pour chaque
    # instrument. M désigne une image mono et S une image stéréo.
    for k in range(d.nbInstruments):
        seq = sorted(
            (d.DD[i][k], i)
            for i, kk in plan
            if kk == k
        )

        txt = ", ".join(
            f"img{i}({'S' if d.TY[i] == 2 else 'M'})@{t}"
            for t, i in seq
        )

        print(f"  Instrument {k + 1} : {txt if txt else '-'}")

    # Contrôle final des contraintes du plan obtenu.
    errors = check_plan(d, plan)

    print(
        "Vérification faisabilité :",
        "OK" if not errors else errors
    )


def run(name, criterion, alpha, verbose=False, cip=None):
    """
    Charge le jeu de données, lance l'optimisation et affiche les résultats.
    """
    d = importlib.import_module(name)

    status, obj, plan = build_and_solve(
        d, criterion, alpha, verbose, cip
    )

    report(name, d, criterion, alpha, status, obj, plan)

    return obj


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter
    )

    ap.add_argument(
        "dataset",
        nargs="?",
        help="Module de données, ex. spotProba4"
    )

    ap.add_argument(
        "--criterion",
        choices=CRITERIA,
        default="choquet"
    )

    ap.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Indice de pessimisme (1 = pessimiste pur, 0 = optimiste pur)"
    )

    ap.add_argument(
        "--all",
        action="store_true",
        help="Exécute tous les jeux de données avec les deux critères"
    )

    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Affiche les messages du solveur"
    )

    ap.add_argument(
        "--cip",
        help="Exporte le modèle au format .cip"
    )

    a = ap.parse_args()

    if a.all:
        # Compare les deux critères sur chacun des cinq jeux de données.
        for name in (
            "spotProba1",
            "spotProba2",
            "spotProba3",
            "spotProba4",
            "spotProba5"
        ):
            for crit in CRITERIA:
                run(name, crit, a.alpha)

    elif a.dataset:
        run(
            a.dataset,
            a.criterion,
            a.alpha,
            a.verbose,
            a.cip
        )

    else:
        ap.error("Spécifiez un jeu de données ou utilisez --all")


if __name__ == "__main__":
    main()