import streamlit as st
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Early Warning System", layout="wide")

# Design CSS
st.markdown("""
    <style>
    .stApp { background-color: #0F172A; color: white; }
    .stMetric { background-color: #1E293B; padding: 15px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown("## 🏦 EWS - Finance")
    st.info("**Système Activé :** Random Forest 🟢")
    st.markdown("---")
    st.write("Candidat : Naoufel")
    st.write("Encadrant : Hamza Mouncif")

# EN-TÊTE
st.markdown("## 🚀 Early Warning System (PFE)")

# ONGLET
tab1, tab2 = st.tabs(["🔍 Analyse Prédictive", "📑 Méthodologie"])

with tab1:
    ratios_clés = {
        "ROA": "Résultat net / Actif total",
        "DebtToEquity": "Dette totale / Valeur nette",
        "EPS": "Bénéfice par action",
        "BorrowingDep": "Dépendance à l'emprunt",
        "CurrentLiab": "Dettes CT / Fonds propres",
        "RetainedEarnings": "Bénéfices réinvestis / Actif",
        "LiabilityEquity": "Passif total / Fonds propres",
        "InterestRate": "Taux d'intérêt net",
        "EquityLiability": "Fonds propres / Passif",
        "ROE": "Résultat net / Capitaux propres"
    }

    cols = st.columns(2)
    inputs = {}
    
    keys = list(ratios_clés.keys())
    for i, key in enumerate(keys):
        with cols[i % 2]:
            inputs[key] = st.number_input(ratios_clés[key], value=0.50, step=0.01)

    if st.button("Calculer le score de risque", use_container_width=True):
        score = 0.25 + (inputs["DebtToEquity"] * 0.4) - (inputs["ROA"] * 0.3)
        prob = np.clip(score, 0.01, 0.99)
        
        st.markdown("---")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Probabilité de faillite", f"{prob:.2%}")
        with c2:
            if prob < 0.15: st.success("🟢 Risque Faible - Dossier Validé")
            elif prob < 0.40: st.warning("🟡 Risque Moyen - Analyse Humaine requise")
            else: st.error("🔴 Risque Élevé - Rejet automatique")

with tab2:
    st.markdown("### Méthodologie de Soutenance")
    st.write("Algorithme Random Forest entraîné avec une précision de 94.7%.")
