import streamlit as st
import joblib

# Charger le modèle et le TF-IDF
model = joblib.load("models/spam_classifier_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Configuration de la page Streamlit
st.set_page_config(page_title="Spam Detector")

st.title("Spam Email Detection App")
st.write("Entrez un texte ci-dessous pour vérifier s'il est spam ou non.")

# Zone de saisie
subject = st.text_input("Email Subject :")
message = st.text_area("Email Message :", height=200)

email_text = subject + ' ' + message

# Prédire
if st.button("Analyser"):
    if email_text.strip() == "":
        st.warning("Veuillez entrer un texte.")
    else:
        email_text = email_text.lower()

        # Transformer le texte avec TF-IDF
        X = vectorizer.transform([email_text])

        # Prédire
        pred = model.predict(X)[0]

        # Affichage du résultat
        if pred == 1:
            st.error("Ce message est **SPAM**.")
        else:
            st.success("Ce message est **NON SPAM**.")
