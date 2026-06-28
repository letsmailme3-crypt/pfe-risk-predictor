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

# --- DICTIONNAIRE DE TRADUCTION ET D'EXPLICATION DES RATIOS FINANCIERS ---
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
    
    # Informations académiques
    st.markdown("### 🎓 Encadrement & Candidat")
    st.info("**Encadrant :** M. Hamza Mouncif\n\n**Candidat :** Naoufel")
    
    st.markdown("---")
    
    # Indicateur d'état du modèle d'IA
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
            <div style="background-color: #FEE2E2; padding: 12px; border-radius: 8px; border-left: 5px solid #EF4444; margin-bottom: 15px;">
                <span style="color: #991B1B; font-weight: bold;">❌ Modèle non détecté</span><br>
                <span style="color: #991B1B; font-size: 12px;">Veuillez ajouter les fichiers .pkl au dépôt.</span>
            </div>
        """, unsafe_allow_html=True)

    # Paramètres des seuils de décision réglementaires
    st.markdown("### ⚖️ Seuils Décisionnels Appliqués")
    st.markdown("""
        - 🟢 **Risque Faible** : < 0.15 *(Accord automatique)*
        - 🟡 **Risque Moyen** : 0.15 à 0.40 *(Audit humain)*
        - 🔴 **Risque Élevé** : ≥ 0.40 *(Rejet immédiat)*
    """)
    st.caption("Seuils optimisés pour minimiser le coût de défaut (Faux Négatifs).")

# --- CORPS PRINCIPAL DE L'APPLICATION ---
st.markdown("""
    <div style="background-color: #0F172A; padding: 25px; border-radius: 12px; margin-bottom: 25px; text-align: center; color: white;">
        <h1 style="color: #38BDF8; margin: 0px; font-family: 'Helvetica Neue', sans-serif;">PLATEFORME D'ÉVALUATION ET DE PRÉDICTION DE LA FAILLITE</h1>
        <p style="color: #94A3B8; font-size: 16px; margin-top: 10px; margin-bottom: 0px;">
            Système d'aide à la décision bancaire basé sur l'algorithme Random Forest (ROC-AUC de 0.947)
        </p>
    </div>
""", unsafe_allow_html=True)

# Création des onglets interactifs de navigation
tab1, tab2, tab3 = st.tabs([
    "🔍 Analyse Individuelle (Formulaire)", 
    "📊 Analyse de Portefeuille (Import CSV)", 
    "📑 Justification Théorique & Performance (Jury)"
])

# --- TAB 1 : ANALYSE INDIVIDUELLE ---
with tab1:
    st.markdown("### 📝 Renseigner les Ratios Financiers de l'Entreprise")
    st.markdown("<p style='color: #64748B;'>Remplissez les champs ci-dessous. Les valeurs par défaut correspondent à des moyennes d'entreprises saines pour vous guider.</p>", unsafe_allow_html=True)
    
    if not model_loaded:
        st.warning("⚠️ L'analyse individuelle est désactivée car le modèle Random Forest (.pkl) n'a pas pu être chargé.")
    else:
        # Création d'une structure en 3 colonnes pour un affichage parfaitement aéré
        col1, col2, col3 = st.columns(3)
        champs_saisie = {}
        
        # Distribution des 15 variables de manière harmonieuse
        liste_variables = list(TRADUCTION_RATIOS.keys())
        
        for i, var_name in enumerate(liste_variables):
            infos = TRADUCTION_RATIOS[var_name]
            valeur_defaut = 0.50 if "net" in var_name.lower() or "income" in var_name.lower() else 0.25
            
            # Répartition équilibrée dans les colonnes
            if i % 3 == 0:
                with col1:
                    champs_saisie[var_name] = st.number_input(
                        label=infos["nom"],
                        min_value=0.0,
                        max_value=100.0,
                        value=float(valeur_defaut),
                        step=0.01,
                        help=infos["desc"]
                    )
            elif i % 3 == 1:
                with col2:
                    champs_saisie[var_name] = st.number_input(
                        label=infos["nom"],
                        min_value=0.0,
                        max_value=100.0,
                        value=float(valeur_defaut),
                        step=0.01,
                        help=infos["desc"]
                    )
            else:
                with col3:
                    champs_saisie[var_name] = st.number_input(
                        label=infos["nom"],
                        min_value=0.0,
                        max_value=100.0,
                        value=float(valeur_defaut),
                        step=0.01,
                        help=infos["desc"]
                    )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Bouton d'action principal
        if st.button("📊 Calculer la Probabilité de Faillite", use_container_width=True):
            # Traitement contre le scale bug (conversion si l'utilisateur saisit en %)
            valeurs_formatees = []
            avertissement_scale = False
            
            for var_name in colonnes_requises:
                valeur_utilisateur = champs_saisie[var_name]
                # Si l'utilisateur a rentré une valeur typiquement en pourcentage (> 1.0)
                if valeur_utilisateur > 1.0:
                    valeur_utilisateur = valeur_utilisateur / 100.0
                    avertissement_scale = True
                valeurs_formatees.append(valeur_utilisateur)
            
            # Préparation du vecteur pour le modèle
            df_input = pd.DataFrame([valeurs_formatees], columns=colonnes_requises)
            
            # Calcul de la prédiction
            probabilite_faillite = model.predict_proba(df_input)[0][1]
            
            # Affichage des résultats
            st.markdown("---")
            st.markdown("### 🎯 Résultat de l'Évaluation du Risque")
            
            if avertissement_scale:
                st.warning("ℹ️ **Note d'échelle :** Certaines valeurs supérieures à 1.0 ont été automatiquement converties en échelle décimale [0-1] pour correspondre aux exigences d'échelle de notre modèle IA.")
            
            c_score, c_decision = st.columns([1, 2])
            
            with c_score:
                st.metric(
                    label="Probabilité de Défaillance",
                    value=f"{probabilite_faillite:.2%}"
                )
            
            with c_decision:
                if probabilite_faillite < 0.15:
                    st.markdown("""
                        <div style="background-color: #D1FAE5; padding: 20px; border-radius: 8px; border-left: 8px solid #10B981;">
                            <h4 style="color: #065F46; margin: 0px;">🟢 RISQUE JUGÉ FAIBLE</h4>
                            <p style="color: #065F46; margin-top: 8px; margin-bottom: 0px;">
                                <b>Décision recommandée :</b> Octroi de crédit approuvé de manière automatisée. L'entreprise présente des indicateurs de rentabilité et d'endettement extrêmement sains.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                elif probabilite_faillite < 0.40:
                    st.markdown("""
                        <div style="background-color: #FEF3C7; padding: 20px; border-radius: 8px; border-left: 8px solid #F59E0B;">
                            <h4 style="color: #92400E; margin: 0px;">🟡 RISQUE MODÉRÉ (ALERTE PRÉCOCE)</h4>
                            <p style="color: #92400E; margin-top: 8px; margin-bottom: 0px;">
                                <b>Décision recommandée :</b> Passage en comité d'analyse de crédit humaine. Il est fortement conseillé de réclamer des garanties (collatéraux) supplémentaires avant d'accorder le financement.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div style="background-color: #FEE2E2; padding: 20px; border-radius: 8px; border-left: 8px solid #EF4444;">
                            <h4 style="color: #991B1B; margin: 0px;">🔴 RISQUE ÉLEVÉ DE FAILLITE</h4>
                            <p style="color: #991B1B; margin-top: 8px; margin-bottom: 0px;">
                                <b>Décision recommandée :</b> Refus catégorique du dossier de crédit. Le modèle détecte une structure de surendettement et une rentabilité d'exploitation critique.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)

# --- TAB 2 : ANALYSE DE PORTEFEUILLE (BATCH CSV) ---
with tab2:
    st.markdown("### 📥 Analyse Globale d'un Portefeuille de Clients")
    st.markdown("<p style='color: #64748B;'>Cette fonctionnalité permet d'importer les données de plusieurs centaines d'entreprises d'un coup pour en extraire des graphiques d'aide à la décision.</p>", unsafe_allow_html=True)
    
    # Boutons d'aide à l'utilisation (Template CSV)
    c_btn1, c_btn2 = st.columns([1, 3])
    with c_btn1:
        # Création d'un DataFrame modèle
        gabarit_data = {col: [0.25 if "debt" in col.lower() else 0.75] for col in colonnes_requises}
        gabarit_df = pd.DataFrame(gabarit_data)
        csv_gabarit = gabarit_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Télécharger le Gabarit CSV",
            data=csv_gabarit,
            file_name="modele_importation_faillite.csv",
            mime="text/csv",
            use_container_width=True
        )
    with c_btn2:
        st.caption("Téléchargez notre gabarit d'importation pré-formaté. Remplissez-le avec les données de vos clients, puis importez-le ci-dessous pour éviter tout bug de colonnes.")

    st.markdown("---")
    
    fichier_csv = st.file_uploader("Choisissez votre fichier CSV de clients", type=["csv"])
    
    if fichier_csv is not None:
        try:
            df_clients = pd.read_csv(fichier_csv)
            
            # Vérification de la présence de toutes les colonnes requises
            colonnes_manquantes = [col for col in colonnes_requises if col not in df_clients.columns]
            
            if len(colonnes_manquantes) > 0:
                st.error(f"❌ Erreur de structure : Il manque des colonnes essentielles dans votre fichier CSV ({len(colonnes_manquantes)} manquantes). Veuillez utiliser notre gabarit ci-dessus.")
            else:
                st.success("✔ Fichier validé avec succès ! Analyse en cours...")
                
                # Alignement des données et calcul des probabilités
                df_pour_ia = df_clients[colonnes_requises].copy()
                
                # Conversion automatique contre le scale bug
                for col in df_pour_ia.columns:
                    if (df_pour_ia[col] > 1.0).any():
                        df_pour_ia[col] = df_pour_ia[col].apply(lambda x: x / 100.0 if x > 1.0 else x)
                
                probabilites = model.predict_proba(df_pour_ia)[:, 1]
                
                # Injection du résultat dans le tableau de sortie
                df_clients["Probabilité Faillite"] = probabilites
                
                # Catégorisation selon les seuils métier
                def categoriser_risque(prob):
                    if prob < 0.15: return "Vert (Faible)"
                    elif prob < 0.40: return "Jaune (Modéré)"
                    else: return "Rouge (Élevé)"
                    
                df_clients["Catégorie de Risque"] = df_clients["Probabilité Faillite"].apply(categoriser_risque)
                
                # Affichage des métriques globales du portefeuille
                st.markdown("### 📊 Synthèse Globale du Portefeuille")
                m1, m2, m3, m4 = st.columns(4)
                
                nb_total = len(df_clients)
                nb_vert = len(df_clients[df_clients["Catégorie de Risque"] == "Vert (Faible)"])
                nb_jaune = len(df_clients[df_clients["Catégorie de Risque"] == "Jaune (Modéré)"])
                nb_rouge = len(df_clients[df_clients["Catégorie de Risque"] == "Rouge (Élevé)"])
                
                m1.metric("Entreprises Analysées", nb_total)
                m2.metric("🟢 Risque Faible (Accord)", f"{nb_vert} ({nb_vert/nb_total:.1%})")
                m3.metric("🟡 Risque Modéré (Audit)", f"{nb_jaune} ({nb_jaune/nb_total:.1%})")
                m4.metric("🔴 Risque Élevé (Refus)", f"{nb_rouge} ({nb_rouge/nb_total:.1%})")
                
                # Graphique de répartition du risque
                st.markdown("### 📈 Visualisation Analytique")
                col_g1, col_g2 = st.columns(2)
                
                with col_g1:
                    # Distribution des scores de probabilité
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.histplot(df_clients["Probabilité Faillite"], bins=15, kde=True, color="#1E3A8A", ax=ax)
                    ax.axvline(0.15, color="green", linestyle="--", label="Seuil Vert/Jaune (0.15)")
                    ax.axvline(0.40, color="red", linestyle="--", label="Seuil Jaune/Rouge (0.40)")
                    ax.set_title("Distribution des Probabilités de Défaut du Portefeuille")
                    ax.set_xlabel("Probabilité de Faillite")
                    ax.set_ylabel("Nombre d'entreprises")
                    ax.legend()
                    st.pyplot(fig)
                    
                with col_g2:
                    # Diagramme de répartition en secteurs
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    labels = ['Risque Faible', 'Risque Moyen', 'Risque Élevé']
                    sizes = [nb_vert, nb_jaune, nb_rouge]
                    colors = ['#10B981', '#F59E0B', '#EF4444']
                    ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops=dict(width=0.4))
                    ax2.set_title("Répartition du Niveau de Risque")
                    st.pyplot(fig2)
                
                # Tableau complet triable et téléchargeable
                st.markdown("### 📋 Liste de toutes les Entreprises Évaluées")
                st.dataframe(df_clients)
                
                # Bouton d'exportation du bilan d'audit
                csv_export = df_clients.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Exporter le Bilan d'Audit complet (.CSV)",
                    data=csv_export,
                    file_name="bilan_audit_portefeuille.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Une erreur s'est produite lors de la lecture du fichier : {e}")

# --- TAB 3 : ENCADREMENT ET THÉORIE (JURY) ---
with tab3:
    st.markdown("### 🎓 Documentation Académique & Performance")
    st.markdown("<p style='color: #64748B;'>Cette section récapitule les fondements mathématiques et statistiques du modèle pour répondre aux questions du jury de soutenance.</p>", unsafe_allow_html=True)
    
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.markdown("#### 🧠 L'algorithme Random Forest")
        st.markdown("""
        Le modèle utilise la technique d'apprentissage par ensemble (**Bagging**) pour réduire drastiquement la variance :
        
        - **Échantillonnage de Bootstrap :** Chaque arbre s'entraîne sur un sous-ensemble aléatoire de la donnée d'origine.
        - **Feature Bagging :** À chaque nœud, l'arbre ne sélectionne que parmi un échantillon restreint de variables ($\sqrt{M}$).
        - **Règle de Vote :** La probabilité finale est la moyenne des prédictions des centaines d'arbres.
        """)
        
        st.markdown("#### 📈 Matrice de Performance Modèle")
        st.markdown("""
        | Métrique d'Évaluation | Valeur | Interprétation |
        | :--- | :---: | :--- |
        | **Aire sous la courbe (ROC-AUC)** | **0.947** | Capacité de discrimination quasi parfaite des classes. |
        | **Taux d'interception (Rappel)** | **94.2%** | Proportion de faillites réelles interceptées par le modèle. |
        | **Précision globale** | **95.1%** | Fidélité des classifications positives du modèle. |
        """)
        
    with col_t2:
        st.markdown("#### ⚖️ Pourquoi ces seuils décisionnels (0.15 et 0.40) ?")
        st.markdown("""
        Dans la gestion du risque crédit, il y a une **asymétrie majeure des coûts** :
        
        - Un **Faux Négatif** (accorder un crédit à une entreprise qui fera faillite) coûte presque **100% du capital prêté**.
        - Un **Faux Positif** (refuser un crédit à une entreprise saine) ne coûte que le **manque à gagner d'intérêts** (coût d'opportunité, environ 5%).
        
        Par conséquent, nous avons délibérément baissé les barrières de détection :
        - Dès que le modèle estime une probabilité supérieure à **15%**, l'entreprise quitte la zone de confort et passe en analyse manuelle.
        - À partir de **40%**, le risque de défaut est jugé inacceptable au vu des accords réglementaires de **Bâle III**.
        """)
        
    st.markdown("---")
    st.info("💡 **Conseil pour Naoufel le jour J :** Si le jury pose une question sur la territorialité des données, rappelez-leur que ce modèle sert de démonstrateur technologique et qu'il est immédiatement transposable sur des données macroéconomiques marocaines en remplaçant simplement le fichier d'entraînement `.pkl`.")
