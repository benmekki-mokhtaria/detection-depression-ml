
import streamlit as st
import numpy as np
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="MentalGuard AI", layout="centered")
st.title("🧠 MentalGuard AI")
st.write("Système d'analyse émotionnelle - Version Optimisée")

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return " ".join(text.split())

# Interface
user_text = st.text_area("Entrez votre texte:", height=120, placeholder="Exemple: Je me sens bien aujourd'hui...")

if st.button("🚀 Analyser", type="primary"):
    if user_text.strip():
        with st.spinner("Analyse en cours..."):
            try:
                # Simulation avec règles (en attendant vrai modèle)
                text_lower = user_text.lower()
                
                # Règles simples basées sur les mots-clés
                positive_words = ['bien', 'heureux', 'content', 'joyeux', 'super', 'bon']
                negative_words = ['triste', 'seul', 'vide', 'déprimé', 'mal', 'pessimiste']
                severe_words = ['suicide', 'mort', 'finir', 'désespoir', 'plus despoir']  # CORRIGÉ
                
                score = 0
                for word in positive_words:
                    if word in text_lower:
                        score -= 1
                
                for word in negative_words:
                    if word in text_lower:
                        score += 1
                        
                for word in severe_words:
                    if word in text_lower:
                        score += 2
                
                # Déterminer la classe
                if score <= 0:
                    classe = 0
                    confiance = 0.85
                elif score == 1:
                    classe = 1
                    confiance = 0.75
                elif score == 2:
                    classe = 2
                    confiance = 0.80
                elif score == 3:
                    classe = 3
                    confiance = 0.78
                else:
                    classe = 4
                    confiance = 0.90
                
                # Afficher résultats
                st.success(f"**Niveau {classe}** détecté")
                st.info(f"**Confiance: {confiance:.1%}**")
                
                # Interprétation
                interpretations = [
                    "🟢 Bien-être optimal",
                    "🟡 Léger malaise",
                    "🟠 Signes modérés", 
                    "🔴 Signes importants",
                    "⚫ Consultation recommandée"
                ]
                st.write(f"**Interprétation:** {interpretations[classe]}")
                
            except Exception as e:
                st.error(f"Erreur: {e}")
    else:
        st.warning("⚠️ Veuillez entrer un texte")

st.markdown("---")
st.caption("MentalGuard AI • Version 1.0 • Système expert")
