# Planification de prises de vue SPOT-5 avec incertitude (nuages + pannes)
# Version complete du fichier spotProbaPartial.py
#
# Usage :
#   python spotProba.py spotProba4                    # maximin (defaut)
#   python spotProba.py spotProba5 --criterion hurwicz --alpha 0.3
#   python spotProba.py spotProba1 --criterion det    # sans incertitude
#   python spotProba.py --all                         # tous les jeux, tous les criteres
#
# Modele (instruments 0,1,2 dans le code = instruments 1,2,3 du sujet)
#   y[i]    = 1 ssi l'image i est selectionnee
#   x[i][k] = 1 ssi l'image i est acquise par l'instrument k
#   mono   : sum_k x[i][k] = y[i]
#   stereo : x[i][0] = x[i][2] = y[i], x[i][1] = 0
#   memoire: sum_i PM[i] y[i] <= PMmax
#   non-chevauchement : x[i][k] + x[j][k] <= 1 si |DD_ik - DD_jk| < DU + |AN_ik - AN_jk| / VI
#   objectif : gain espere pessimiste (maximin), optimiste (maximax), Hurwicz, ou deterministe

import argparse
import importlib
from itertools import combinations

from pyscipopt import Model, quicksum

CRITERIA = ("det", "maximin", "maximax", "hurwicz")
STEREO_INSTRUMENTS = (0, 2)  # instruments avant et arriere


def cloud_probability(d, criterion, alpha):
    """Probabilite de nuages p_i retenue pour chaque image selon le critere.
    L'esperance de gain etant decroissante en p_i, le pire cas est ProbaSup,
    le meilleur cas est ProbaInf, Hurwicz interpole (reste lineaire)."""
    n = d.nbImages
    if criterion == "maximin":
        return [d.ProbaSup[i] for i in range(n)]
    if criterion == "maximax":
        return [d.ProbaInf[i] for i in range(n)]
    if criterion == "hurwicz":
        return [alpha * d.ProbaSup[i] + (1 - alpha) * d.ProbaInf[i] for i in range(n)]
    return [0.0] * n


def in_conflict(d, i, j, k):
    """Deux acquisitions i, j sur l'instrument k sont incompatibles si l'ecart entre
    leurs dates est inferieur a la duree d'acquisition plus le temps de rotation.
    Ecrit sans division : |DDi - DDj| * VI < DU * VI + |ANi - ANj|."""
    return abs(d.DD[i][k] - d.DD[j][k]) * d.VI < d.DU * d.VI + abs(d.AN[i][k] - d.AN[j][k])


def gain_coefficients(d, criterion, alpha):
    """Coefficients c[i][k] (mono, sur x) et c[i] (stereo, sur y) de l'objectif."""
    p = cloud_probability(d, criterion, alpha)
    use_failure = criterion != "det"
    f = d.Failure if use_failure else [0.0] * d.nbInstruments
    c_mono, c_stereo = {}, {}
    for i in range(d.nbImages):
        if d.TY[i] == 1:
            for k in range(d.nbInstruments):
                c_mono[i, k] = d.PA[i] * (1 - p[i]) * (1 - f[k])
        else:
            c_stereo[i] = d.PA[i] * (1 - p[i]) * (1 - f[0]) * (1 - f[2])
    return c_mono, c_stereo


def build_and_solve(d, criterion="maximin", alpha=0.5, verbose=False, cip=None):
    n, m = d.nbImages, d.nbInstruments
    model = Model()
    if not verbose:
        model.hideOutput()

    # variables
    y = {i: model.addVar(vtype="B", name=f"select{i}") for i in range(n)}
    x = {(i, k): model.addVar(vtype="B", name=f"assignto{i}_{k}")
         for i in range(n) for k in range(m)}

    # objectif
    c_mono, c_stereo = gain_coefficients(d, criterion, alpha)
    model.setObjective(
        quicksum(c * x[i, k] for (i, k), c in c_mono.items())
        + quicksum(c * y[i] for i, c in c_stereo.items()),
        sense="maximize",
    )

    # selection / affectation
    for i in range(n):
        if d.TY[i] == 1:
            model.addCons(quicksum(x[i, k] for k in range(m)) == y[i])
        else:
            for k in range(m):
                if k in STEREO_INSTRUMENTS:
                    model.addCons(x[i, k] == y[i])
                else:
                    model.addCons(x[i, k] == 0)

    # memoire
    model.addCons(quicksum(d.PM[i] * y[i] for i in range(n)) <= d.PMmax)

    # non-chevauchement, instrument par instrument
    for k in range(m):
        for i, j in combinations(range(n), 2):
            # un stereo n'utilise jamais l'instrument 2 (sa date DD y vaut 0, sans sens)
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
    """Verification independante de la faisabilite d'un plan (liste de (image, instrument))."""
    errors = []
    by_image = {}
    for i, k in plan:
        by_image.setdefault(i, []).append(k)
    for i, ks in by_image.items():
        if d.TY[i] == 1 and len(ks) != 1:
            errors.append(f"mono {i} sur {ks}")
        if d.TY[i] == 2 and sorted(ks) != list(STEREO_INSTRUMENTS):
            errors.append(f"stereo {i} sur {ks}")
    if sum(d.PM[i] for i in by_image) > d.PMmax:
        errors.append("memoire depassee")
    for k in range(d.nbInstruments):
        imgs = sorted((d.DD[i][k], i) for i, kk in plan if kk == k)
        for (t1, i), (t2, j) in zip(imgs, imgs[1:]):
            need = d.DU + abs(d.AN[i][k] - d.AN[j][k]) / d.VI
            if t2 - t1 < need - 1e-9:
                errors.append(f"instrument {k}: images {i},{j} ecart {t2 - t1} < {need}")
    return errors


def expected_gain(d, plan, p):
    """Gain espere d'un plan pour des probabilites de nuages p_i donnees."""
    g = 0.0
    for i, k in plan:
        if d.TY[i] == 1:
            g += d.PA[i] * (1 - p[i]) * (1 - d.Failure[k])
        elif k == STEREO_INSTRUMENTS[0]:
            g += d.PA[i] * (1 - p[i]) * (1 - d.Failure[0]) * (1 - d.Failure[2])
    return g


def report(name, d, criterion, alpha, status, obj, plan):
    label = criterion + (f" (alpha={alpha})" if criterion == "hurwicz" else "")
    print(f"\n=== {name} | critere {label} | statut {status} ===")
    if status != "optimal":
        return
    print(f"valeur de l'objectif : {obj:.4f}")
    lo = expected_gain(d, plan, d.ProbaSup)
    hi = expected_gain(d, plan, d.ProbaInf)
    print(f"gain espere du plan  : [{lo:.4f}, {hi:.4f}]  (nuages ProbaSup / ProbaInf)")
    selected = sorted({i for i, _ in plan})
    print(f"images selectionnees : {len(selected)}/{d.nbImages}, "
          f"memoire {sum(d.PM[i] for i in selected)}/{d.PMmax}")
    for k in range(d.nbInstruments):
        seq = sorted((d.DD[i][k], i) for i, kk in plan if kk == k)
        txt = ", ".join(f"img{i}({'S' if d.TY[i] == 2 else 'M'})@{t}" for t, i in seq)
        print(f"  instrument {k + 1} : {txt if txt else '-'}")
    errors = check_plan(d, plan)
    print("verification faisabilite :", "OK" if not errors else errors)


def run(name, criterion, alpha, verbose=False, cip=None):
    d = importlib.import_module(name)
    status, obj, plan = build_and_solve(d, criterion, alpha, verbose, cip)
    report(name, d, criterion, alpha, status, obj, plan)
    return obj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("dataset", nargs="?", help="module de donnees, ex. spotProba4")
    ap.add_argument("--criterion", choices=CRITERIA, default="maximin")
    ap.add_argument("--alpha", type=float, default=0.5, help="parametre de Hurwicz (poids du pire cas)")
    ap.add_argument("--all", action="store_true", help="tous les jeux et tous les criteres")
    ap.add_argument("--verbose", action="store_true", help="log du solveur")
    ap.add_argument("--cip", help="ecrire le probleme au format .cip")
    a = ap.parse_args()

    if a.all:
        for name in ("spotProba1", "spotProba2", "spotProba3", "spotProba4", "spotProba5"):
            for crit in CRITERIA:
                run(name, crit, a.alpha)
    elif a.dataset:
        run(a.dataset, a.criterion, a.alpha, a.verbose, a.cip)
    else:
        ap.error("donner un jeu de donnees ou --all")


if __name__ == "__main__":
    main()
