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

import spotProba as sp

DATASETS = ["spotProba1", "spotProba2", "spotProba3", "spotProba4", "spotProba5"]

st.set_page_config(page_title="SPOT-5 : plan d'acquisition sous incertitude", layout="wide")
st.title("SPOT-5 — plan d'acquisition sous incertitude")
st.caption(
    "Modele MILP (selection + affectation + non-chevauchement), gain espere pessimiste, "
    "moyen ou optimiste selon le parametre de Hurwicz."
)

# ------------------------------------------------------------------
# barre laterale
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Parametres")
    dataset_name = st.selectbox("Jeu de donnees", DATASETS, index=3)
    alpha = st.slider(
        "alpha (Hurwicz)", 0.0, 1.0, 1.0, 0.05,
        help="alpha=1 : pessimiste (p=ProbaSup). alpha=0 : optimiste (p=ProbaInf). "
             "alpha=0.5 : decideur moyen.",
    )
    st.markdown("---")
    show_curve = st.checkbox("Courbe de sensibilite (gain optimal vs alpha)", value=True)
    show_robustness = st.checkbox("Prix de la robustesse (deterministe vs maximin)", value=True)
    n_points = st.slider("Points de la courbe", 5, 41, 21, disabled=not show_curve)


@st.cache_resource
def load_dataset(name):
    return importlib.import_module(name)


@st.cache_data(show_spinner=False)
def solve(name, alpha):
    d = load_dataset(name)
    status, obj, plan = sp.build_and_solve(d, criterion="hurwicz", alpha=alpha)
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
col2.metric("Gain (objectif Hurwicz)", f"{obj:.2f}" if obj is not None else "-")

if status == "optimal":
    lo = sp.expected_gain(d, plan, d.ProbaSup)   # pire cas reel du plan
    hi = sp.expected_gain(d, plan, d.ProbaInf)   # meilleur cas reel du plan
    col3.metric("Gain espere, pire cas", f"{lo:.2f}")
    col4.metric("Gain espere, meilleur cas", f"{hi:.2f}")

    selected = sorted({i for i, _ in plan})
    st.write(
        f"**{len(selected)} / {d.nbImages} images selectionnees** — "
        f"memoire {sum(d.PM[i] for i in selected)} / {d.PMmax}"
    )

    errors = sp.check_plan(d, plan)
    if errors:
        st.error(f"Plan infaisable (verification independante) : {errors}")
    else:
        st.success("Plan verifie faisable (memoire, affectation, non-chevauchement).")

    # tableau du plan
    rows = []
    for i, k in sorted(plan, key=lambda t: (t[1], d.DD[t[0]][t[1]])):
        rows.append({
            "Image": i,
            "Type": "stereo" if d.TY[i] == 2 else "mono",
            "Instrument": k + 1,
            "Debut": d.DD[i][k],
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
    ax.set_xlabel("temps (s depuis debut de revolution)")
    st.pyplot(fig)
else:
    st.error("Pas de solution optimale trouvee pour ce jeu de parametres.")

# ------------------------------------------------------------------
# prix de la robustesse
# ------------------------------------------------------------------
if show_robustness:
    st.subheader("Prix de la robustesse")
    det_status, det_obj, det_plan = solve_det(dataset_name)
    if det_status == "optimal" and status == "optimal":
        det_worst = sp.expected_gain(d, det_plan, d.ProbaSup)
        maximin_worst = sp.expected_gain(d, plan, d.ProbaSup)
        df = pd.DataFrame({
            "Plan": ["Deterministe (ignore l'incertitude)", f"Hurwicz alpha={alpha}"],
            "Gain nominal (sans incertitude)": [det_obj, sum(d.PA[i] for i in {i for i, _ in plan})],
            "Gain reel en pire cas (ProbaSup)": [det_worst, maximin_worst],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(
            "Le plan deterministe maximise le gain sans tenir compte des nuages ni des pannes ; "
            "evalue en pire cas reel, il peut perdre significativement face au plan qui integre "
            "l'incertitude des le calcul du plan."
        )

# ------------------------------------------------------------------
# courbe de sensibilite
# ------------------------------------------------------------------
if show_curve:
    st.subheader("Sensibilite du gain optimal a alpha")
    alphas = [i / (n_points - 1) for i in range(n_points)]
    with st.spinner("Resolution pour chaque valeur de alpha..."):
        gains = []
        for a in alphas:
            _, o, _ = sp.build_and_solve(d, criterion="hurwicz", alpha=a)
            gains.append(o)
    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    ax2.plot(alphas, gains, marker="o", markersize=3, color="#4C72B0")
    ax2.axvline(alpha, color="grey", linestyle="--", linewidth=1)
    ax2.set_xlabel("alpha  (1 = pessimiste, 0 = optimiste)")
    ax2.set_ylabel("gain optimal de l'objectif Hurwicz")
    ax2.set_title(f"{dataset_name}")
    st.pyplot(fig2)
    st.caption(
        "Le gain optimal decroit avec alpha : plus le decideur est prudent (alpha proche de 1), "
        "moins il compte sur les images incertaines et plus son gain garanti est bas."
    )
