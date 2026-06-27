import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Configuration de la page de l'application
st.set_page_config(
    page_title="Risk Predictor - Bank Early Warning System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS pour une belle interface moderne
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        color: #1E3A8A;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .risk-low {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #10B981;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #F59E0B;
        font-weight: bold;
    }
    .risk-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #EF4444;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Seuils de décision validés sur votre notebook (0.15 et 0.40)
SEUIL_FAIBLE_MOYEN = 0.15
SEUIL_MOYEN_ELEVE = 0.40

# Les 15 ratios financiers clés identifiés par votre Random Forest
KEY_FEATURES = [
    'Continuous interest rate (after tax)',
    'Total debt/Total net worth',
    'Persistent EPS in the Last Four Seasons',
    'After-tax net Interest Rate',
    'Borrowing dependency',
    'Net Income to Total Assets',
    'Retained Earnings to Total Assets',
    'Liability to Equity',
    'Per Share Net profit before tax (Yuan \u00a5)',
    'Pre-tax net Interest Rate',
    'Equity to Liability',
    'Net Income to Stockholder\'s Equity',
    'ROA(B) before interest and depreciation after tax',
    'Current Liability to Equity',
    'Net profit before tax/Paid-in capital'
]

# Chargement du modèle .pkl et des colonnes de données
@st.cache_resource
def charger_fichiers():
    modele_path = 'modele_faillite_rf.pkl'
    colonnes_path = 'colonnes_modele.pkl'
    
    if os.path.exists(modele_path) and os.path.exists(colonnes_path):
        model = joblib.load(modele_path)
        columns = joblib.load(colonnes_path)
        is_fallback = False
    else:
        # Modèle de secours temporaire si lancé sans les fichiers .pkl
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_dummy = pd.DataFrame(np.random.rand(10, len(KEY_FEATURES)), columns=KEY_FEATURES)
        y_dummy = np.random.choice([0, 1], size=10)
        model.fit(X_dummy, y_dummy)
        columns = KEY_FEATURES
        is_fallback = True
    return model, columns, is_fallback

model, columns, is_fallback = charger_fichiers()

# Barre latérale (Sidebar) de l'application
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank.png", width=70)
    st.title("Early Warning System")
    st.markdown("### PFE - Analyse du Risque de Crédit")
    st.markdown("**Encadrant :** M. Hamza Mouncif")
    st.markdown("---")
    
    if is_fallback:
        st.warning("⚠️ Fichiers d'export .pkl non détectés dans ce dossier.")
    else:
        st.success("✅ Modèle Random Forest optimisé chargé avec succès.")
        
    st.info(f"""
    **Seuils décisionnels appliqués :**
    * 🟢 **Risque Faible** : $< {SEUIL_FAIBLE_MOYEN:.2f}$
    * 🟡 **Risque Moyen** : de ${SEUIL_FAIBLE_MOYEN:.2f}$ à ${SEUIL_MOYEN_ELEVE:.2f}$
    * 🔴 **Risque Élevé** : $\ge {SEUIL_MOYEN_ELEVE:.2f}$
    """)

# En-tête de la page principale
st.markdown('<div class="main-title">PLATEFORME D\'ÉVALUATION DU RISQUE DE FAILLITE</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Système d\'aide à la décision bancaire basé sur les résultats de notre modèle d\'apprentissage.</div>', unsafe_allow_html=True)

# Onglets d'utilisation
tab1, tab2 = st.tabs(["🔍 Analyse Individuelle (Formulaire)", "📊 Analyse de Portefeuille (Fichier CSV)"])

# --- ONGLET 1 : FORMULAIRE INDIVIDUEL ---
with tab1:
    st.subheader("Veuillez renseigner les ratios financiers de l'entreprise :")
    
    cols = st.columns(3)
    inputs = {}
    
    # Valeurs moyennes de référence par défaut
    valeurs_defaut = {
        'Continuous interest rate (after tax)': 0.78,
        'Total debt/Total net worth': 0.02,
        'Persistent EPS in the Last Four Seasons': 0.22,
        'After-tax net Interest Rate': 0.80,
        'Borrowing dependency': 0.37,
        'Net Income to Total Assets': 0.80,
        'Retained Earnings to Total Assets': 0.93,
        'Liability to Equity': 0.28,
        'Per Share Net profit before tax (Yuan \u00a5)': 0.18,
        'Pre-tax net Interest Rate': 0.80,
        'Equity to Liability': 0.02,
        'Net Income to Stockholder\'s Equity': 0.83,
        'ROA(B) before interest and depreciation after tax': 0.46,
        'Current Liability to Equity': 0.33,
        'Net profit before tax/Paid-in capital': 0.18
    }
    
    # Génération automatique des cases de saisie
    for i, feature in enumerate(KEY_FEATURES):
        col_courante = cols[i % 3]
        with col_courante:
            inputs[feature] = st.number_input(
                label=feature,
                min_value=0.0,
                max_value=1.0,
                value=valeurs_defaut.get(feature, 0.50),
                step=0.01
            )
            
    st.markdown("---")
    
    if st.button("Calculer le risque de faillite", use_container_width=True):
        row_dict = {}
        for col in columns:
            if col in inputs:
                row_dict[col] = inputs[col]
            else:
                row_dict[col] = 0.0  # Remplissage par défaut pour les variables secondaires
                
        input_df = pd.DataFrame([row_dict])
        
        # Prédiction de la probabilité de faillite
        proba = model.predict_proba(input_df)[0, 1]
        
        # Affichage du résultat
        st.subheader("Résultat du diagnostic bancaire :")
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.metric(
                label="Probabilité de faillite estimée",
                value=f"{proba * 100:.2f} %"
            )
            
        with c2:
            if proba < SEUIL_FAIBLE_MOYEN:
                st.markdown(f"""
                <div class="risk-low">
                    🟢 RISQUE FAIBLE (Probabilité : {proba*100:.1f}%)<br>
                    Décision : Dossier Approuvé<br>
                    <span style='font-weight:normal; font-size:0.9rem;'>
                    Les ratios de solvabilité et de rentabilité sont excellents. Recommandation d'octroi de crédit automatique.
                    </span>
                </div>
                """, unsafe_allow_html=True)
            elif proba < SEUIL_MOYEN_ELEVE:
                st.markdown(f"""
                <div class="risk-medium">
                    🟡 RISQUE MOYEN (Probabilité : {proba*100:.1f}%)<br>
                    Décision : Examen Manuel Requis<br>
                    <span style='font-weight:normal; font-size:0.9rem;'>
                    Le profil présente quelques signes de fragilité ou un niveau d'endettement modéré. Des garanties complémentaires sont à négocier.
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-high">
                    🔴 RISQUE ÉLEVÉ (Probabilité : {proba*100:.1f}%)<br>
                    Décision : Refus Systématique<br>
                    <span style='font-weight:normal; font-size:0.9rem;'>
                    Risque de défaut critique dépassant la tolérance aux pertes de l'institution. Dossier écarté.
                    </span>
                </div>
                """, unsafe_allow_html=True)

# --- ONGLET 2 : ANALYSE DE PORTEFEUILLE (CSV) ---
with tab2:
    st.subheader("Traitement de données en volume (Portefeuille)")
    st.write("Importez un fichier CSV contenant les ratios de plusieurs entreprises pour les classifier instantanément.")
    
    fichier_charge = st.file_uploader("Fichier CSV", type=["csv"])
    
    if fichier_charge is not None:
        try:
            df_portefeuille = pd.read_csv(fichier_charge)
            st.success("Données chargées !")
            
            # Reconstruction des colonnes attendues pour le modèle
            df_eval = df_portefeuille.copy()
            for col in columns:
                if col not in df_eval.columns:
                    df_eval[col] = 0.0
            df_eval = df_eval[columns]
            
            # Prédictions
            probabilites = model.predict_proba(df_eval)[:, 1]
            df_portefeuille['Probabilité Faillite'] = probabilites
            
            # Classification
            def classifier(p):
                if p < SEUIL_FAIBLE_MOYEN: return 'Faible'
                elif p < SEUIL_MOYEN_ELEVE: return 'Moyen'
                else: return 'Élevé'
            df_portefeuille['Classe de Risque'] = df_portefeuille['Probabilité Faillite'].apply(classifier)
            
            # Statistique globale
            repartition = df_portefeuille['Classe de Risque'].value_counts()
            
            ca, cb, cc = st.columns(3)
            with ca: st.metric("Dossiers sains (Faible)", f"{repartition.get('Faible', 0)}")
            with cb: st.metric("Dossiers en surveillance (Moyen)", f"{repartition.get('Moyen', 0)}")
            with cc: st.metric("Dossiers rejetés (Élevé)", f"{repartition.get('Élevé', 0)}")
            
            # Graphique de répartition
            st.write("---")
            st.subheader("Répartition visuelle du portefeuille")
            fig, ax = plt.subplots(figsize=(8, 3.5))
            palette_couleurs = {'Faible': '#10B981', 'Moyen': '#F59E0B', 'Élevé': '#EF4444'}
            sns.countplot(data=df_portefeuille, x='Classe de Risque', order=['Faible', 'Moyen', 'Élevé'], palette=palette_couleurs, ax=ax)
            ax.set_title("Volume de dossiers par catégorie")
            st.pyplot(fig)
            
            # Table d'affichage
            st.subheader("Données prédites :")
            st.dataframe(df_portefeuille[['Classe de Risque', 'Probabilité Faillite'] + [c for c in KEY_FEATURES if c in df_portefeuille.columns]].head(100))
            
        except Exception as err:
            st.error(f"Une erreur est survenue lors de l'analyse : {err}")
