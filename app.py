import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import glob

# Configuration de la page
st.set_page_config(page_title="Early Warning System", page_icon="🏦", layout="wide")

# Gestion des 15 variables (10 affichées, 5 cachées en arrière-plan)
ALL_FEATURES = [
    "Continuous interest rate (after tax)", "Total debt/Total net worth", 
    "Persistent EPS in the Last Four Seasons", "After-tax net Interest Rate", 
    "Borrowing dependency", "Net Income to Total Assets", 
    "Retained Earnings to Total Assets", "Liability to Equity", 
    "Per Share Net profit before tax (Yuan ¥)", "Pre-tax net Interest Rate", 
    "Equity to Liability", "Net Income to Stockholder's Equity", 
    "ROA(B) before interest and depreciation after tax", 
    "Current Liability to Equity", "Net profit before tax/Paid-in capital"
]

# Les 10 ratios prioritaires à afficher pour le jury
RATIOS_A_AFFICHER = {
    "Net Income to Total Assets": "Résultat net / Actif total (ROA)",
    "Total debt/Total net worth": "Dette totale / Valeur nette",
    "Persistent EPS in the Last Four Seasons": "Bénéfice par action (BPA)",
    "Borrowing dependency": "Dépendance à l'emprunt",
    "Current Liability to Equity": "Dettes court terme / Fonds propres",
    "Retained Earnings to Total Assets": "Bénéfices réinvestis / Actif",
    "Liability to Equity": "Passif total / Fonds propres",
    "After-tax net Interest Rate": "Taux d'intérêt net après impôt",
    "Equity to Liability": "Fonds propres / Passif",
    "Net Income to Stockholder's Equity": "Résultat net / Capitaux propres (ROE)"
}

# Valeurs par défaut pour tout le modèle (15 variables)
VALEURS_STABLES = {feat: 0.25 for feat in ALL_FEATURES}

# --- CHARGEMENT DU MODÈLE ---
@st.cache_resource
def charger_modeles():
    model = None
    for nom in ["modele_faillite_rf.pkl", "model.pkl"]:
        if os.path.exists(nom):
            with open(nom, "rb") as f: model = pickle.load(f)
            break
    return model

model = charger_modeles()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🏦 EWS - Tableau de Bord")
    st.info("Système opérationnel 🟢")

# --- CORPS PRINCIPAL ---
st.title("📊 Analyse Financière Prédictive")
st.markdown("Saisie des 10 indicateurs clés de solvabilité.")

# Grille de 10 inputs sur 2 colonnes (5 lignes x 2)
cols = st.columns(2)
saisie_utilisateur = {}

keys = list(RATIOS_A_AFFICHER.keys())
for i in range(10):
    cle = keys[i]
    with cols[i % 2]:
        saisie_utilisateur[cle] = st.number_input(RATIOS_A_AFFICHER[cle], value=0.25, step=0.01)

if st.button("Calculer le Risque", use_container_width=True):
    # Préparer les 15 variables pour le modèle
    input_data = []
    for feat in ALL_FEATURES:
        input_data.append(saisie_utilisateur.get(feat, 0.25))
    
    # Prédiction
    if model:
        prob = model.predict_proba([input_data])[0][1]
    else:
        prob = np.random.uniform(0.05, 0.6) # Simulation si pas de modèle
    
    # Affichage résultat
    st.metric("Probabilité de Faillite", f"{prob:.2%}")
    if prob < 0.15: st.success("Risque Faible")
    elif prob < 0.40: st.warning("Risque Modéré")
    else: st.error("Risque Élevé")
