

import argparse
import importlib
from itertools import combinations

from pyscipopt import Model, quicksum

CRITERIA = ("det", "choquet")

STEREO_INSTRUMENTS = (0, 2)


def get_choquet_capacity_for_clouds(d, criterion, alpha):
    
    n = d.nbImages

    if criterion == "choquet":
        return [
            alpha * d.ProbaSup[i] + (1 - alpha) * d.ProbaInf[i]
            for i in range(n)
        ]

    return [0.0] * n


def in_conflict(d, i, j, k):
  
    return (
        abs(d.DD[i][k] - d.DD[j][k]) * d.VI
        < d.DU * d.VI + abs(d.AN[i][k] - d.AN[j][k])
    )


def gain_coefficients(d, criterion, alpha):
  
    p = get_choquet_capacity_for_clouds(d, criterion, alpha)

    use_failure = criterion != "det"
    f = d.Failure if use_failure else [0.0] * d.nbInstruments

    c_mono, c_stereo = {}, {}

    for i in range(d.nbImages):
        if d.TY[i] == 1:

            for k in range(d.nbInstruments):
                c_mono[i, k] = (
                    d.PA[i]
                    * (1 - p[i])
                    * (1 - f[k])
                )
        else:

            c_stereo[i] = (
                d.PA[i]
                * (1 - p[i])
                * (1 - f[0])
                * (1 - f[2])
            )

    return c_mono, c_stereo


def build_and_solve(d, criterion="choquet", alpha=1.0,
                    verbose=False, cip=None):
  
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
            model.addCons(
                quicksum(x[i, k] for k in range(m)) == y[i]
            )
        else:
            for k in range(m):
                if k in STEREO_INSTRUMENTS:
                    model.addCons(x[i, k] == y[i])
                else:
                    model.addCons(x[i, k] == 0)

    model.addCons(
        quicksum(d.PM[i] * y[i] for i in range(n)) <= d.PMmax
    )

    for k in range(m):
        for i, j in combinations(range(n), 2):

            if (
                (d.TY[i] == 2 or d.TY[j] == 2)
                and k not in STEREO_INSTRUMENTS
            ):
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

    objective = model.getObjVal() if status == "optimal" else None

    return status, objective, plan


def check_plan(d, plan):
   
    errors = []
    by_image = {}

    for i, k in plan:
        by_image.setdefault(i, []).append(k)

    for i, ks in by_image.items():
        if d.TY[i] == 1 and len(ks) != 1:
            errors.append(f"Erreur Mono : {i} affectée sur {ks}")

        if (
            d.TY[i] == 2
            and sorted(ks) != list(STEREO_INSTRUMENTS)
        ):
            errors.append(f"Erreur Stéréo : {i} affectée sur {ks}")

    if sum(d.PM[i] for i in by_image) > d.PMmax:
        errors.append("Erreur : La capacité mémoire totale est dépassée.")

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
    
    g = 0.0

    for i, k in plan:
        if d.TY[i] == 1:

            g += (
                d.PA[i]
                * (1 - p[i])
                * (1 - d.Failure[k])
            )

        elif k == STEREO_INSTRUMENTS[0]:

            g += (
                d.PA[i]
                * (1 - p[i])
                * (1 - d.Failure[0])
                * (1 - d.Failure[2])
            )

    return g


def report(name, d, criterion, alpha, status, obj, plan):
    
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

    errors = check_plan(d, plan)

    print(
        "Vérification faisabilité :",
        "OK" if not errors else errors
    )


def run(name, criterion, alpha, verbose=False, cip=None):

 
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