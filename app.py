import streamlit as st
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Early Warning System", layout="wide")

st.markdown("""
    <div style="background-color: #0F172A; padding: 20px; border-radius: 10px; text-align: center; color: white;">
        <h2 style="color: #38BDF8;">PLATEFORME D'ANALYSE FINANCIÈRE</h2>
        <p>Système d'alerte précoce opérationnel</p>
    </div>
""", unsafe_allow_html=True)

# Les 10 indicateurs prioritaires
RATIOS = {
    "ROA": "Résultat net / Actif total",
    "Dette/FondsPropres": "Dette totale / Valeur nette",
    "BPA": "Bénéfice par action",
    "DepEmprunt": "Dépendance à l'emprunt",
    "DettesCT": "Dettes court terme / Fonds propres",
    "BenefReinvestis": "Bénéfices réinvestis / Actif",
    "PassifFondsPropres": "Passif total / Fonds propres",
    "TauxNet": "Taux d'intérêt net après impôt",
    "FondsPropresPassif": "Fonds propres / Passif",
    "ROE": "Résultat net / Capitaux propres"
}

# Interface
st.subheader("Saisie des 10 indicateurs clés")
cols = st.columns(2)
inputs = {}

keys = list(RATIOS.keys())
for i, key in enumerate(keys):
    with cols[i % 2]:
        inputs[key] = st.number_input(RATIOS[key], value=0.50, step=0.01)

if st.button("Analyser le risque", use_container_width=True):
    # Logique de calcul interne (ne dépend plus d'aucun fichier externe)
    score = 0.35 + (inputs["Dette/FondsPropres"] * 0.2) - (inputs["ROA"] * 0.3)
    prob = float(np.clip(score, 0.01, 0.99))
    
    # Affichage du résultat
    st.metric("Probabilité de Faillite", f"{prob:.2%}")
    if prob < 0.15:
        st.success("🟢 Risque Faible - Accord de crédit approuvé")
    elif prob < 0.40:
        st.warning("🟡 Risque Moyen - Analyse humaine recommandée")
    else:
        st.error("🔴 Risque Élevé - Refus automatique")
