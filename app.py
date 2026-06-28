import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration de la page Streamlit pour un rendu moderne et responsive
st.set_page_config(
    page_title="Risk Predictor - Bank Early Warning System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DICTIONNAIRE DE TRADUCTION SÉCURISÉ ---
TRADUCTION_RATIOS = {
    "Continuous interest rate (after tax)": {
        "nom": "Taux d'intérêt continu (après impôt)",
        "desc": "Mesure la rentabilité nette des capitaux par rapport aux charges d'intérêts continues après déduction fiscale. Un taux stable indique une bonne santé financière."
    },
    "Total debt/Total net worth": {
        "nom": "Dette totale / Valeur nette globale",
        "desc": "Ratio d'endettement mesurant la proportion de dettes par rapport aux capitaux propres. Un ratio supérieur à 1.0 indique que la dette dépasse l'actif net."
    },
    "Persistent EPS in the Last Four Seasons": {
        "nom": "Bénéfice par action (BPA) persistant",
        "desc": "Bénéfice net consolidé par action sur les 4 derniers trimestres. Un BPA positif et stable est un puissant indicateur de non-faillite."
    },
    "After-tax net Interest Rate": {
        "nom": "Taux d'intérêt net après impôt",
        "desc": "Ratio mesurant la marge d'intérêt nette de l'entreprise après prise en compte des obligations fiscales."
    },
    "Borrowing dependency": {
        "nom": "Dépendance à l'emprunt financier",
        "desc": "Mesure à quel point l'actif de l'entreprise est financé par de la dette extérieure. Plus ce ratio est élevé, plus le risque de solvabilité augmente."
    },
    "Net Income to Total Assets": {
        "nom": "Résultat net / Actif total (ROA)",
        "desc": "Mesure l'efficacité avec laquelle l'entreprise utilise l'ensemble de ses actifs pour générer du profit."
    },
    "Retained Earnings to Total Assets": {
        "nom": "Bénéfices non distribués / Actif total",
        "desc": "Part des bénéfices réinvestis dans l'entreprise. Un ratio élevé démontre une capacité historique d'autofinancement solide."
    },
    "Liability to Equity": {
        "nom": "Total du passif / Capitaux propres",
        "desc": "Compare l'ensemble des obligations financières dues aux tiers par rapport aux fonds apportés par les actionnaires."
    },
    "Per Share Net profit before tax (Yuan ¥)": {
        "nom": "Bénéfice net avant impôt par action",
        "desc": "Indique la rentabilité opérationnelle par action de l'entreprise avant l'impact des politiques fiscales."
    },
    "Pre-tax net Interest Rate": {
        "nom": "Taux d'intérêt net avant impôt",
        "desc": "Rendement financier généré par les activités opérationnelles avant déduction des charges d'impôt."
    },
    "Equity to Liability": {
        "nom": "Capitaux propres / Total du passif",
        "desc": "Inverse du ratio d'endettement. Plus il est élevé, plus la couverture des créanciers par les fonds propres est sécurisée."
    },
    "Net Income to Stockholder's Equity": {
        "nom": "Résultat net / Capitaux propres (ROE)",
        "desc": "Mesure la rentabilité des capitaux investis par les actionnaires de l'entreprise."
    },
    "ROA(B) before interest and depreciation after tax": {
        "nom": "ROA(B) avant intérêts & amortissements (après impôt)",
        "desc": "Indicateur de la performance économique brute de l'outil de production de l'entreprise."
    },
    "Current Liability to Equity": {
        "nom": "Dettes à court terme / Capitaux propres",
        "desc": "Mesure la vulnérabilité financière à court terme face aux fonds propres disponibles."
    },
    "Net profit before tax/Paid-in capital": {
        "nom": "Bénéfice net avant impôt / Capital libéré",
        "desc": "Rendement généré par rapport au capital social initialement versé par les associés."
    }
}

# --- CHARGEMENT DU MODÈLE ET DES COLONNES ---
@st.cache_resource
def charger_fichiers_modeles():
    try:
        with open("modele_faillite_rf.pkl", "rb") as f_model:
            model = pickle.load(f_model)
        with open("colonnes_modele.pkl", "rb") as f_cols:
            colonnes = pickle.load(f_cols)
        return model, colonnes, True
    except Exception as e:
        return None, None, False

model, colonnes_requises, model_loaded = charger_fichiers_modeles()

# SÉCURITÉ : Si le modèle n'est pas chargé, on utilise les 15 colonnes par défaut pour éviter de planter !
if not model_loaded or colonnes_requises is None:
    colonnes_requises = list(TRADUCTION_RATIOS.keys())

# --- DESIGN DE LA BARRE LATÉRALE ---
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 10px 0px;">
            <span style="font-size: 55px;">🏦</span>
            <h2 style="margin-top: 10px; color: #1E3A8A; font-family: 'Helvetica Neue', sans-serif;">Early Warning System</h2>
            <p style="color: #6B7280; font-size: 14px;">PFE - Analyse Prédictive du Risque de Crédit</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🎓 Encadrement & Candidat")
    st.info("**Encadrant :** M. Hamza Mouncif\n\n**Candidat :** Naoufel")
    
    st.markdown("---")
    
    st.markdown("### 🧠 État du Modèle IA")
    if model_loaded:
        st.markdown("""
            <div style="background-color: #D1FAE5; padding: 12px; border-radius: 8px; border-left: 5px solid #10B981; margin-bottom: 15px;">
                <span style="color: #065F46; font-weight: bold;">✔ Random Forest Optimisé</span><br>
                <span style="color: #065F46; font-size: 12px;">Modèle chargé avec succès.</span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="background-color: #FEF3C7; padding: 12px; border-radius: 8px; border-left: 5px solid #F59E0B; margin-bottom: 15px;">
                <span style="color: #92400E; font-weight: bold;">⚠️ Mode Démonstration</span><br>
                <span style="color: #92400E; font-size: 12px;">Fichiers .pkl non détectés. L'interface reste active pour le jury.</span>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("### ⚖️ Seuils Décisionnels Appliqués")
    st.markdown("""
        - 🟢 **Risque Faible** : < 0.15 *(Accord automatique)*
        - 🟡 **Risque Moyen** : 0.15 à 0.40 *(Audit humain)*
        - 🔴 **Risque Élevé** : ≥ 0.40 *(Rejet immédiat)*
    """)
    st.caption("Seuils optimisés pour minimiser le coût de défaut (Faux Négatifs).")

# --- CORPS PRINCIPAL ---
st.markdown("""
    <div style="background-color: #0F172A; padding: 25px; border-radius: 12px; margin-bottom: 25px; text-align: center; color: white;">
        <h1 style="color: #38BDF8; margin: 0px; font-family: 'Helvetica Neue', sans-serif;">PLATEFORME D'ÉVALUATION ET DE PRÉDICTION DE LA FAILLITE</h1>
        <p style="color: #94A3B8; font-size: 16px; margin-top: 10px; margin-bottom: 0px;">
            Système d'aide à la décision bancaire basé sur l'algorithme Random Forest (ROC-AUC de 0.947)
        </p>
    </div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "🔍 Analyse Individuelle (Formulaire)", 
    "📊 Analyse de Portefeuille (Import CSV)", 
    "📑 Justification Théorique & Performance (Jury)"
])

# --- TAB 1 : ANALYSE INDIVIDUELLE ---
with tab1:
    st.markdown("### 📝 Renseigner les Ratios Financiers de l'Entreprise")
    st.markdown("<p style='color: #64748B;'>Remplissez les champs ci-dessous. Les valeurs par défaut correspondent à des moyennes d'entreprises saines.</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    champs_saisie = {}
    
    for i, var_originale in enumerate(colonnes_requises):
        if var_originale in TRADUCTION_RATIOS:
            label_affiche = TRADUCTION_RATIOS[var_originale]["nom"]
            aide_texte = TRADUCTION_RATIOS[var_originale]["desc"]
        else:
            label_affiche = var_originale
            aide_texte = "Ratio financier requis pour l'analyse."
            
        valeur_defaut = 0.50 if "net" in var_originale.lower() or "income" in var_originale.lower() else 0.25
        
        if i % 3 == 0:
            with col1:
                champs_saisie[var_originale] = st.number_input(
                    label=label_affiche, min_value=0.0, max_value=100.0,
                    value=float(valeur_defaut), step=0.01, help=aide_texte, key=f"ind_{i}"
                )
        elif i % 3 == 1:
            with col2:
                champs_saisie[var_originale] = st.number_input(
                    label=label_affiche, min_value=0.0, max_value=100.0,
                    value=float(valeur_defaut), step=0.01, help=aide_texte, key=f"ind_{i}"
                )
        else:
            with col3:
                champs_saisie[var_originale] = st.number_input(
                    label=label_affiche, min_value=0.0, max_value=100.0,
                    value=float(valeur_defaut), step=0.01, help=aide_texte, key=f"ind_{i}"
                )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("📊 Calculer la Probabilité de Faillite", use_container_width=True):
        if not model_loaded:
            st.error("❌ Impossible de calculer la probabilité réelle : les fichiers de modèle (.pkl) ne sont pas détectés dans votre dossier GitHub.")
        else:
            valeurs_formatees = []
            avertissement_scale = False
            
            for var_name in colonnes_requises:
                valeur_utilisateur = champs_saisie[var_name]
                if valeur_utilisateur > 1.0:
                    valeur_utilisateur = valeur_utilisateur / 100.0
                    avertissement_scale = True
                valeurs_formatees.append(valeur_utilisateur)
            
            df_input = pd.DataFrame([valeurs_formatees], columns=colonnes_requises)
            probabilite_faillite = model.predict_proba(df_input)[0][1]
            
            st.markdown("---")
            st.markdown("### 🎯 Résultat de l'Évaluation du Risque")
            
            if avertissement_scale:
                st.info("ℹ️ **Correction d'échelle appliquée :** Les pourcentages ont été convertis à l'échelle décimale [0-1] pour correspondre au modèle.")
                
            c_score, c_decision = st.columns([1, 2])
            with c_score:
                st.metric(label="Probabilité de Défaillance", value=f"{probabilite_faillite:.2%}")
                
            with c_decision:
                if probabilite_faillite < 0.15:
                    st.markdown("""
                        <div style="background-color: #D1FAE5; padding: 20px; border-radius: 8px; border-left: 8px solid #10B981;">
                            <h4 style="color: #065F46; margin: 0px;">🟢 RISQUE JUGÉ FAIBLE</h4>
                            <p style="color: #065F46; margin-top: 8px; margin-bottom: 0px;">
                                <b>Décision :</b> Accord automatique du crédit. Les indicateurs sont excellents.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                elif probabilite_faillite < 0.40:
                    st.markdown("""
                        <div style="background-color: #FEF3C7; padding: 20px; border-radius: 8px; border-left: 8px solid #F59E0B;">
                            <h4 style="color: #92400E; margin: 0px;">🟡 RISQUE MODÉRÉ (ALERTE PRÉCOCE)</h4>
                            <p style="color: #92400E; margin-top: 8px; margin-bottom: 0px;">
                                <b>Décision :</b> Transmission pour audit manuel. Exiger des garanties de second rang.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div style="background-color: #FEE2E2; padding: 20px; border-radius: 8px; border-left: 8px solid #EF4444;">
                            <h4 style="color: #991B1B; margin: 0px;">🔴 RISQUE ÉLEVÉ DE FAILLITE</h4>
                            <p style="color: #991B1B; margin-top: 8px; margin-bottom: 0px;">
                                <b>Décision :</b> Refus automatique du dossier. Risque d'impayé inacceptable.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)

# --- TAB 2 : ANALYSE EN PORTEFEUILLE ---
with tab2:
    st.markdown("### 📥 Analyse Globale d'un Portefeuille de Clients")
    
    c_btn1, c_btn2 = st.columns([1, 3])
    with c_btn1:
        gabarit_data = {col: [0.25 if "debt" in col.lower() or "liability" in col.lower() else 0.75] for col in colonnes_requises}
        gabarit_df = pd.DataFrame(gabarit_data)
        csv_gabarit = gabarit_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Télécharger le Gabarit CSV",
            data=csv_gabarit,
            file_name="gabarit_import_credits.csv",
            mime="text/csv",
            use_container_width=True
        )
    with c_btn2:
        st.caption("Téléchargez le gabarit ci-contre, remplissez les lignes avec vos données réelles, puis déposez-le ci-dessous.")

    st.markdown("---")
    fichier_csv = st.file_uploader("Importer le fichier CSV complété", type=["csv"])
    
    if fichier_csv is not None:
        if not model_loaded:
            st.error("❌ L'analyse par lot nécessite que le modèle IA soit correctement chargé sur GitHub (fichiers .pkl détectés).")
        else:
            try:
                df_clients = pd.read_csv(fichier_csv)
                colonnes_manquantes = [col for col in colonnes_requises if col not in df_clients.columns]
                
                if len(colonnes_manquantes) > 0:
                    st.error(f"❌ Erreur : Il manque {len(colonnes_manquantes)} colonnes requises dans le fichier. Utilisez le gabarit téléchargeable ci-dessus.")
                else:
                    st.success("✔ Structure de fichier valide !")
                    df_pour_ia = df_clients[colonnes_requises].copy()
                    
                    for col in df_pour_ia.columns:
                        if (df_pour_ia[col] > 1.0).any():
                            df_pour_ia[col] = df_pour_ia[col].apply(lambda x: x / 100.0 if x > 1.0 else x)
                    
                    probabilites = model.predict_proba(df_pour_ia)[:, 1]
                    df_clients["Probabilité Faillite"] = probabilites
                    
                    def attribuer_classe(p):
                        if p < 0.15: return "Vert (Faible)"
                        elif p < 0.40: return "Jaune (Modéré)"
                        else: return "Rouge (Élevé)"
                    df_clients["Niveau de Risque"] = df_clients["Probabilité Faillite"].apply(attribuer_classe)
                    
                    m1, m2, m3, m4 = st.columns(4)
                    total_obs = len(df_clients)
                    n_vert = len(df_clients[df_clients["Niveau de Risque"] == "Vert (Faible)"])
                    n_jaune = len(df_clients[df_clients["Niveau de Risque"] == "Jaune (Modéré)"])
                    n_rouge = len(df_clients[df_clients["Niveau de Risque"] == "Rouge (Élevé)"])
                    
                    m1.metric("Entreprises Audités", total_obs)
                    m2.metric("🟢 Risque Faible", f"{n_vert} ({n_vert/total_obs:.1%})")
                    m3.metric("🟡 Risque Moyen", f"{n_jaune} ({n_jaune/total_obs:.1%})")
                    m4.metric("🔴 Risque Élevé", f"{n_rouge} ({n_rouge/total_obs:.1%})")
                    
                    col_g1, col_g2 = st.columns(2)
                    with col_g1:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.histplot(df_clients["Probabilité Faillite"], bins=15, kde=True, color="#1E3A8A", ax=ax)
                        ax.axvline(0.15, color="green", linestyle="--")
                        ax.axvline(0.40, color="red", linestyle="--")
                        ax.set_title("Distribution des Risques du Portefeuille")
                        st.pyplot(fig)
                    with col_g2:
                        fig2, ax2 = plt.subplots(figsize=(6, 4))
                        ax2.pie([n_vert, n_jaune, n_rouge], labels=['Faible', 'Moyen', 'Élevé'], 
                                autopct='%1.1f%%', startangle=90, colors=['#10B981', '#F59E0B', '#EF4444'], wedgeprops=dict(width=0.4))
                        ax2.set_title("Répartition des Décisions d'Octroi")
                        st.pyplot(fig2)
                    
                    st.dataframe(df_clients)
                    
                    csv_export = df_clients.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Télécharger le Rapport d'Audit (.CSV)",
                        data=csv_export,
                        file_name="rapport_complet_portefeuille.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Erreur d'analyse : {e}")

# --- TAB 3 : JURY & THÉORIE ---
with tab3:
    st.markdown("### 🎓 Justifications Scientifiques")
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.markdown("#### 🧠 Robustesse du Random Forest")
        st.markdown("""
        L'algorithme des forêts aléatoires est particulièrement résistant au surapprentissage (*overfitting*) grâce au double principe d'échantillonnage de Bootstrap et de sélection aléatoire des variables au niveau de chaque nœud :
        
        - **Biais/Variance optimal :** L'agrégation des prédictions réduit considérablement la variance globale.
        - **Évaluation Out-Of-Bag (OOB) :** Permet une validation interne robuste tout au long du processus.
        """)
        
        st.markdown("#### 📈 Indicateurs Validés par Apprentissage")
        st.markdown("""
        - **ROC-AUC (Aire sous la courbe) :** **0.947** (Excellent pouvoir séparateur).
        - **Rappel (Sensibilité) :** **94.2%** des faillites réelles interceptées à temps.
        """)
    
    with col_t2:
        st.markdown("#### ⚖️ Rationalisation des Seuils Financiers")
        st.markdown("""
        - **Coût d'erreur asymétrique :** Accorder un crédit à une entreprise insolvable engendre une perte brute sèche pour l'établissement financier. Refuser par erreur un crédit sain ne cause qu'un coût d'opportunité égal à la marge d'intérêt manquée.
        - **Seuils d'Alerte :**
          - **`< 0.15` (Vert) :** Zone de sécurité absolue.
          - **`0.15 - 0.40` (Jaune) :** Zone grise exigeant une intervention de l'analyste de crédit et un dépôt de collatéral.
          - **`≥ 0.40` (Rouge) :** Alerte critique, blocage réglementaire.
        """)
