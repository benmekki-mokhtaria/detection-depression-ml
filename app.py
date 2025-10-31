
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from sklearn.metrics import classification_report
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configuration
st.set_page_config(
    page_title="Détection de Dépression - Modèle ML",
    page_icon="🧠",
    layout="wide"
)

# Style CSS professionnel
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .metric-box {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem;
        text-align: center;
    }
    .result-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.3rem;
        border: 3px solid;
    }
    .low-risk { 
        background-color: #d4edda; 
        color: #155724;
        border-color: #c3e6cb;
    }
    .medium-risk { 
        background-color: #fff3cd; 
        color: #856404;
        border-color: #ffeaa7;
    }
    .high-risk { 
        background-color: #f8d7da; 
        color: #721c24;
        border-color: #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-title">🧠 Détection de Dépression - Modèle ML Professionnel</h1>', unsafe_allow_html=True)

# Charger le modèle et les données
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        top_words = pd.read_csv('top_depression_words.csv')
        with open('model_performance.json', 'r') as f:
            performance = json.load(f)
        return model, vectorizer, top_words, performance
    except Exception as e:
        st.error(f"Erreur chargement modèle: {e}")
        return None, None, None, None

# Fonction de nettoyage (identique à l'entraînement)
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    try:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
        return ' '.join(words)
    except:
        return text

# Chargement
model, vectorizer, top_words, performance = load_model()

if model is None:
    st.error("❌ Impossible de charger le modèle. Vérifiez les fichiers.")
else:
    # Navigation
    page = st.sidebar.radio("📊 Navigation", 
                           ["🏠 Accueil", "🔍 Détection ML", "📈 Performances", "🔑 Mots Importants"])
    
    if page == "🏠 Accueil":
        st.header("🎯 Présentation du Modèle ML")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **🤖 Modèle Machine Learning :**
            - Regression Logistique
            - Vectorisation TF-IDF
            - Entraîné sur 6000 posts Reddit
            - 85% de précision globale
            """)
        
        with col2:
            st.success("""
            **📊 Dataset :**
            - 1202 posts de dépression
            - 4755 posts autres
            - Textes nettoyés et prétraités
            - Labels vérifiés
            """)
        
        st.markdown("---")
        st.subheader("🚀 Fonctionnalités Avancées")
        st.write("""
        Cette application utilise le **vrai modèle ML entraîné** pour :
        - Analyser le texte avec l'intelligence artificielle
        - Donner un score de confiance probabiliste
        - Identifier les mots les plus prédictifs
        - Fournir une analyse détaillée
        """)
    
    elif page == "🔍 Détection ML":
        st.header("🔍 Détection par Modèle ML")
        st.write("**Analyse de texte avec intelligence artificielle**")
        
        # Zone de texte
        user_text = st.text_area(
            "Entrez le texte à analyser :",
            placeholder="Exemple: 'I feel so sad and hopeless these days, I cant sleep and everything seems empty...'",
            height=150
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_btn = st.button("🤖 Analyser avec le Modèle ML", type="primary", use_container_width=True)
        
        if analyze_btn:
            if not user_text.strip():
                st.error("❌ Veuillez entrer un texte à analyser")
            else:
                with st.spinner("🔬 Analyse en cours par le modèle ML..."):
                    # Nettoyage du texte
                    cleaned_text = clean_text(user_text)
                    
                    # Vectorisation
                    text_vector = vectorizer.transform([cleaned_text])
                    
                    # Prédiction
                    prediction = model.predict(text_vector)[0]
                    probabilities = model.predict_proba(text_vector)[0]
                    
                    # Score de confiance
                    confidence = probabilities[1] if prediction == 1 else probabilities[0]
                    
                    # Analyse des mots importants dans le texte
                    user_words = cleaned_text.split()
                    important_detected = []
                    for word in user_words:
                        if word in top_words['word'].values:
                            importance = top_words[top_words['word'] == word]['coefficient'].values[0]
                            important_detected.append((word, importance))
                    
                    # Trier par importance
                    important_detected.sort(key=lambda x: x[1], reverse=True)
                    
                    # Affichage des résultats
                    st.subheader("📊 Résultats du Modèle ML")
                    
                    # Box de résultat
                    if prediction == 1:
                        risk_class = "high-risk"
                        risk_level = "DÉPRESSION DÉTECTÉE"
                        emoji = "🔴"
                        confidence_pct = probabilities[1] * 100
                    else:
                        risk_class = "low-risk" 
                        risk_level = "PAS DE DÉPRESSION"
                        emoji = "✅"
                        confidence_pct = probabilities[0] * 100
                    
                    st.markdown(f'<div class="result-box {risk_class}">', unsafe_allow_html=True)
                    st.markdown(f'<h2>{emoji} {risk_level}</h2>', unsafe_allow_html=True)
                    st.markdown(f'<h3>Confiance du modèle: {confidence_pct:.1f}%</h3>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Métriques détaillées
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Prédiction", "Dépression" if prediction == 1 else "Normal")
                    
                    with col2:
                        st.metric("Confiance", f"{confidence_pct:.1f}%")
                    
                    with col3:
                        prob_depression = probabilities[1] * 100
                        st.metric("Probabilité Dépression", f"{prob_depression:.1f}%")
                    
                    with col4:
                        st.metric("Mots importants détectés", len(important_detected))
                    
                    # Mots importants détectés
                    if important_detected:
                        st.subheader("🔍 Mots Importants Détectés")
                        words_df = pd.DataFrame(important_detected, columns=['Mot', 'Coefficient'])
                        st.dataframe(words_df.head(10), use_container_width=True)
                        
                        # Graphique des mots importants
                        fig, ax = plt.subplots(figsize=(10, 6))
                        top_5 = words_df.head(5)
                        colors = ['red' if coef > 0 else 'green' for coef in top_5['Coefficient']]
                        ax.barh(top_5['Mot'], top_5['Coefficient'], color=colors)
                        ax.set_title('Top 5 Mots les Plus Importants (Coefficients du Modèle)')
                        ax.set_xlabel('Coefficient (Importance)')
                        st.pyplot(fig)
                    
                    # Recommandations
                    st.subheader("💡 Analyse et Recommandations")
                    
                    if prediction == 1:
                        st.error(f"""
                        **🔴 Le modèle a détecté des signes de dépression avec {confidence_pct:.1f}% de confiance.**
                        
                        **Recommandations :**
                        - Consultez un professionnel de santé mentale
                        - Parlez à des personnes de confiance
                        - Ligne d'écoute: 3114 (France)
                        - Ne restez pas isolé(e)
                        """)
                    else:
                        st.success(f"""
                        **✅ Le modèle n'a pas détecté de signes de dépression ({confidence_pct:.1f}% de confiance).**
                        
                        **Conseils de bien-être :**
                        - Continuez à prendre soin de votre santé mentale
                        - Maintenez une routine équilibrée
                        - Restez connecté avec vos proches
                        """)
    
    elif page == "📈 Performances":
        st.header("📈 Performances du Modèle")
        
        if performance:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Accuracy", f"{performance['accuracy']:.1%}")
            
            with col2:
                st.metric("Précision Dépression", f"{performance['precision_depression']:.1%}")
            
            with col3:
                st.metric("Rappel Dépression", f"{performance['recall_depression']:.1%}")
            
            # Graphique de performance
            st.subheader("📊 Métriques de Performance")
            metrics = ['Accuracy', 'Précision', 'Rappel']
            values = [performance['accuracy'], performance['precision_depression'], performance['recall_depression']]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(metrics, values, color=['blue', 'green', 'orange'])
            ax.set_ylim(0, 1)
            ax.set_ylabel('Score')
            ax.set_title('Performance du Modèle de Classification')
            
            # Ajouter les valeurs sur les barres
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{value:.1%}', ha='center', va='bottom')
            
            st.pyplot(fig)
            
            st.info("""
            **📝 Interprétation :**
            - **Accuracy** : Pourcentage global de bonnes prédictions
            - **Précision** : Parmi les prédits "dépression", combien le sont vraiment
            - **Rappel** : Parmi les vrais "dépression", combien sont détectés
            """)
    
    elif page == "🔑 Mots Importants":
        st.header("🔑 Mots les Plus Importants")
        st.write("**Analyse des coefficients du modèle de régression logistique**")
        
        if top_words is not None:
            # Top 20 mots dépression
            top_20_depression = top_words.head(20)
            top_20_other = top_words.tail(20)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Mots Associés à la Dépression")
                st.dataframe(top_20_depression, use_container_width=True)
                
                # Graphique mots dépression
                fig1, ax1 = plt.subplots(figsize=(10, 8))
                ax1.barh(top_20_depression['word'], top_20_depression['coefficient'], color='red')
                ax1.set_title('Top 20 Mots - Dépression')
                ax1.set_xlabel('Coefficient (Positif = Dépression)')
                st.pyplot(fig1)
            
            with col2:
                st.subheader("📉 Mots Associés au Non-Dépression")
                st.dataframe(top_20_other, use_container_width=True)
                
                # Graphique mots non-dépression
                fig2, ax2 = plt.subplots(figsize=(10, 8))
                ax2.barh(top_20_other['word'], top_20_other['coefficient'], color='green')
                ax2.set_title('Top 20 Mots - Non-Dépression')
                ax2.set_xlabel('Coefficient (Négatif = Normal)')
                st.pyplot(fig2)
            
            st.info("""
            **💡 Explication :**
            - **Coefficient positif** : Le mot est associé à la dépression
            - **Coefficient négatif** : Le mot est associé à l'absence de dépression
            - Ces coefficients viennent de l'analyse réelle des données Reddit
            """)

# Pied de page professionnel
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p>🧠 <em>Modèle ML Professionnel - Détection de Dépression - Accuracy: 85%</em></p>
</div>
""", unsafe_allow_html=True)
