# SPOT-5 — Optimisation du plan d'acquisition

Modèle MILP (sélection + affectation + non-chevauchement) pour le case study SPOT-5
(M2 IA et Décision, Toulouse III). Résout le choix des images à acquérir et de
l'instrument à utiliser, avec prise en compte optionnelle des nuages et des pannes
d'instrument.

## Contenu

| Fichier | Rôle |
|---|---|
| `spotProba.py` | Modèle et résolution (pyscipopt) |
| `spotProba1.py` … `spotProba5.py` | Jeux de données fournis |
| `app_streamlit.py` | Interface interactive (α, plan, courbes) |
| `spot5_modele.tex` | Présentation Beamer du modèle mathématique |

## Installation

```bash
pip install pyscipopt pandas matplotlib streamlit
```

## Utilisation en ligne de commande

```bash
python spotProba.py spotProba4                       # critère maximin (défaut)
python spotProba.py spotProba5 --criterion hurwicz --alpha 0.3
python spotProba.py spotProba1 --criterion det        # sans incertitude
python spotProbaPartial.py --all                             # 5 jeux x 4 critères
```

Options :
- `--criterion` : `det`, `maximin`, `maximax`, `hurwicz`
- `--alpha` : poids du pire cas pour Hurwicz (1 = pessimiste, 0 = optimiste, 0.5 = moyen)
- `--verbose` : affiche le log du solveur
- `--cip FICHIER` : exporte le problème au format `.cip`

Chaque exécution affiche le plan par instrument, l'intervalle de gain espéré réel
du plan ([ProbaSup, ProbaInf]), et une vérification de faisabilité indépendante du
solveur (mémoire, affectation, non-chevauchement).

## Interface interactive

```bash
streamlit run streamlit.py
```

Permet de choisir le jeu de données et α, et affiche en direct : le plan, la frise
par instrument, le prix de la robustesse (plan déterministe vs maximin évalués en
pire cas réel), et la courbe de sensibilité du gain optimal en fonction de α.

## Modèle

**Variables** : `y[i]` (sélection), `x[i][k]` (affectation de l'image i à
l'instrument k, k ∈ {0,1,2} = avant/nadir/arrière).

**Contraintes**
- Mono (`TY=1`) : `Σ_k x[i][k] = y[i]`
- Stéréo (`TY=2`) : `x[i][0] = x[i][2] = y[i]`, `x[i][1] = 0`
- Mémoire : `Σ_i PM[i]·y[i] ≤ PMmax`
- Non-chevauchement, par instrument : `x[i][k] + x[j][k] ≤ 1` si
  `|DD[i][k] − DD[j][k]| < DU + |AN[i][k] − AN[j][k]| / VI`

**Objectif**
- Sans incertitude : `max Σ PA[i]·y[i]`
- Avec incertitude, critère de Hurwicz (pessimiste si α=1) :
  `p[i] = α·ProbaSup[i] + (1−α)·ProbaInf[i]`
  - mono sur k : `PA[i]·(1−p[i])·(1−Failure[k])`
  - stéréo : `PA[i]·(1−p[i])·(1−Failure[0])·(1−Failure[2])`

Le critère maximin (α=1) correspond à l'intégrale de Choquet de la capacité
inférieure induite par `[ProbaInf, ProbaSup]` ; l'additivité du gain la réduit à
une somme d'espérances inférieures marginales (voir `spot5_modele.tex`).

## Résultats de référence (validés)

| Jeu | Gain optimal sans incertitude |
|---|---|
| spotProba1 | 70 |
| spotProba2 | 60 |
| spotProba3 | 60 |

| Jeu | maximin | moyen (α=0.5) | maximax |
|---|---|---|---|
| spotProba4 | 333 | 362 | 400 |
| spotProba5 | 553.65 | 615.88 | 678.12 |
