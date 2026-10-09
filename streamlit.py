# Interface Streamlit pour explorer le modele SPOT-5 sous incertitude.
#
# Installation :
#   pip install streamlit pyscipopt pandas matplotlib
#
# Lancement (depuis le dossier contenant spotProba.py, spotProba1..5.py) :
#   streamlit run app_streamlit.py

import importlib

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import spotProbaPartial as sp

DATASETS = ["spotProba1", "spotProba2", "spotProba3", "spotProba4", "spotProba5"]

st.set_page_config(page_title="SPOT-5 : plan d'acquisition sous incertitude", layout="wide")
st.title("SPOT-5 — plan d'acquisition sous incertitude")
st.caption(
    "Modèle MILP (sélection + affectation + non-chevauchement). "
    "Le gain espéré est évalué par l'Intégrale de Choquet paramétrée par alpha "
    "(modélisant l'attitude face aux probabilités imprécises de nuages)."
)

# ------------------------------------------------------------------
# barre laterale
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Paramètres")
    dataset_name = st.selectbox("Jeu de données", DATASETS, index=3)
    alpha = st.slider(
        "alpha (Aversion à l'ambiguïté)", 0.0, 1.0, 1.0, 0.05,
        help="alpha=1 : Choquet pessimiste (utilise ProbaSup, équivalent Bel). "
             "alpha=0 : Choquet optimiste (utilise ProbaInf, équivalent Pl). "
             "alpha=0.5 : Décideur neutre face à l'ambiguïté.",
    )
    st.markdown("---")
    show_curve = st.checkbox("Courbe de sensibilité (gain optimal vs alpha)", value=True)
    show_robustness = st.checkbox("Prix de la robustesse (déterministe vs Choquet)", value=True)
    n_points = st.slider("Points de la courbe", 5, 41, 21, disabled=not show_curve)


@st.cache_resource
def load_dataset(name):
    return importlib.import_module(name)


@st.cache_data(show_spinner=False)
def solve(name, alpha):
    d = load_dataset(name)
    # Remplacement de "hurwicz" par le critère unifié "choquet"
    status, obj, plan = sp.build_and_solve(d, criterion="choquet", alpha=alpha)
    return status, obj, plan


@st.cache_data(show_spinner=False)
def solve_det(name):
    d = load_dataset(name)
    status, obj, plan = sp.build_and_solve(d, criterion="det")
    return status, obj, plan


d = load_dataset(dataset_name)
status, obj, plan = solve(dataset_name, alpha)

# ------------------------------------------------------------------
# resultat principal
# ------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Statut", status)
col2.metric("Gain (Objectif Choquet)", f"{obj:.2f}" if obj is not None else "-")

if status == "optimal":
    lo = sp.expected_gain(d, plan, d.ProbaSup)   # pire cas reel du plan
    hi = sp.expected_gain(d, plan, d.ProbaInf)   # meilleur cas reel du plan
    col3.metric("Gain espéré, pire cas", f"{lo:.2f}")
    col4.metric("Gain espéré, meilleur cas", f"{hi:.2f}")

    selected = sorted({i for i, _ in plan})
    st.write(
        f"**{len(selected)} / {d.nbImages} images sélectionnées** — "
        f"mémoire {sum(d.PM[i] for i in selected)} / {d.PMmax}"
    )

    errors = sp.check_plan(d, plan)
    if errors:
        st.error(f"Plan infaisable (vérification indépendante) : {errors}")
    else:
        st.success("Plan vérifié faisable (mémoire, affectation, non-chevauchement).")

    # tableau du plan
    rows = []
    for i, k in sorted(plan, key=lambda t: (t[1], d.DD[t[0]][t[1]])):
        rows.append({
            "Image": i,
            "Type": "stéréo" if d.TY[i] == 2 else "mono",
            "Instrument": k + 1,
            "Début": d.DD[i][k],
            "Angle": d.AN[i][k],
            "Gain PA_i": d.PA[i],
            "ProbaSup nuages": d.ProbaSup[i],
        })
    st.subheader("Plan d'acquisition")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # frise par instrument
    st.subheader("Frise des acquisitions par instrument")
    fig, ax = plt.subplots(figsize=(9, 2.2))
    colors = {1: "#4C72B0", 2: "#55A868", 3: "#C44E52"}
    for i, k in plan:
        t0 = d.DD[i][k]
        ax.broken_barh([(t0, d.DU)], (k * 10, 8), facecolors=colors[k + 1])
        ax.text(t0, k * 10 + 4, str(i), fontsize=7, va="center", ha="left", color="white")
    ax.set_yticks([4, 14, 24])
    ax.set_yticklabels(["Instrument 1", "Instrument 2", "Instrument 3"])
    ax.set_xlabel("temps (s depuis début de révolution)")
    st.pyplot(fig)
else:
    st.error("Pas de solution optimale trouvée pour ce jeu de paramètres.")

# ------------------------------------------------------------------
# prix de la robustesse
# ------------------------------------------------------------------
if show_robustness:
    st.subheader("Prix de la robustesse")
    det_status, det_obj, det_plan = solve_det(dataset_name)
    if det_status == "optimal" and status == "optimal":
        det_worst = sp.expected_gain(d, det_plan, d.ProbaSup)
        choquet_worst = sp.expected_gain(d, plan, d.ProbaSup)
        df = pd.DataFrame({
            "Plan": ["Déterministe (ignore l'incertitude)", f"Choquet paramétré (alpha={alpha})"],
            "Gain nominal (sans incertitude)": [det_obj, sum(d.PA[i] for i in {i for i, _ in plan})],
            "Gain réel en pire cas (ProbaSup)": [det_worst, choquet_worst],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(
            "Le plan déterministe maximise le gain sans tenir compte des probabilités de nuages ni des pannes. "
            "Évalué dans la réalité (en pire cas météo), il subit de lourdes pertes face au plan robuste qui "
            "intègre l'incertitude via l'intégrale de Choquet dès l'optimisation."
        )

# ------------------------------------------------------------------
# courbe de sensibilite
# ------------------------------------------------------------------
if show_curve:
    st.subheader("Sensibilité du gain optimal à alpha")
    alphas = [i / (n_points - 1) for i in range(n_points)]
    with st.spinner("Résolution MILP pour chaque valeur de alpha..."):
        gains = []
        for a in alphas:
            _, o, _ = sp.build_and_solve(d, criterion="choquet", alpha=a)
            gains.append(o)
    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    ax2.plot(alphas, gains, marker="o", markersize=3, color="#4C72B0")
    ax2.axvline(alpha, color="grey", linestyle="--", linewidth=1)
    ax2.set_xlabel("alpha (1 = pessimiste pur / Bel, 0 = optimiste pur / Pl)")
    ax2.set_ylabel("Gain optimal (Intégrale de Choquet)")
    ax2.set_title(f"{dataset_name}")
    st.pyplot(fig2)
    st.caption(
        "L'utilité espérée garantie décroît avec alpha : plus le décideur est averse à l'ambiguïté "
        "(alpha proche de 1), plus il s'appuie sur la probabilité inférieure de succès, réduisant "
        "mécaniquement l'espérance calculée par l'intégrale de Choquet."
    )