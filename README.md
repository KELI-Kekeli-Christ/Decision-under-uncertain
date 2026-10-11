# Optimisation du plan d’acquisition du satellite SPOT-5

Ce projet vise à maximiser le gain des images acquises par le satellite SPOT-5, en tenant compte des contraintes de mémoire, de disponibilité des instruments, des conflits temporels et des incertitudes liées aux nuages et aux pannes.

## Fonctionnalités

- **Optimisation MILP** avec SCIP et `pyscipopt`.
- **Gestion des incertitudes** : intégrale de Choquet pour les probabilités imprécises de couverture nuageuse et espérance pour les pannes.
- **Analyse de sensibilité** grâce au paramètre `alpha`, du scénario pessimiste au scénario optimiste.
- **Interface interactive** avec Streamlit pour visualiser les plans d’acquisition et les gains.

## Installation

Installer SCIP, puis les dépendances Python :

```bash
pip install pyscipopt streamlit pandas matplotlib
```

## Utilisation

Lancer le tableau de bord :

```bash
streamlit run app_streamlit.py
```

Exécuter le modèle :

```bash
python spotProbaPartial.py spotProba5 --criterion choquet --alpha 0.5
```

Traiter tous les jeux de données :

```bash
python spotProbaPartial.py --all
```

## Démonstration

- **Code source** : [GitHub](https://github.com/KELI-Kekeli-Christ/Decision-under-uncertain)
- **Application interactive** : [aidecision.streamlit.app](https://aidecision.streamlit.app/)

*L’application peut prendre quelques secondes à démarrer après une période d’inactivité.*

## Auteurs

- AKLI Melissa
- OUATTARA Aminata Kampanougou
- KELI Kekeli Christ

*Master 2 IAFA — Université de Toulouse*
