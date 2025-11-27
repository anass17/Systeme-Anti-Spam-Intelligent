# Systeme-Anti-Spam-Intelligent

Un projet complet de machine learning et NLP (Natural Language Processing) pour classifier les emails en **spam** ou **ham** grâce au prétraitement de texte, la vectorisation TF-IDF et plusieurs algorithmes de classification. Le modèle final est déployé avec **Streamlit**.

---

## Vue d'ensemble du projet

Ce projet construit un système intelligent capable de détecter les emails spam avec une grande précision. Il comprend :

* Exploration et nettoyage du dataset
* Prétraitement du texte (tokenisation, suppression des stopwords, stemming...)
* Vectorisation avec **TF-IDF**
* Entraînement et optimisation de plusieurs modèles ML
* Sélection et sauvegarde du meilleur modèle
* Déploiement du modèle avec **Streamlit**

---

## Structure du projet

```
📁 Systeme-Anti-Spam-Intelligent
│
├── 📄 requirements.txt                     # Dépendances
├── 📄 README.md                            # Documentation du projet
├── 📁 app/                     
├    └── 📄 streamlit_app.py                # Application Streamlit
├── 📁 data/                    
├    ├── 📁 raw/                            # Données brutes         
├    └── 📁 processed/                      # Données propres
├── 📁 models/                              # Documentation du projet
├    ├── 📄 spam_classifier_model.pkl       # Modèle ML (SVC) sauvegardé
├    └── 📄 tfidf_vectorizer.pkl            # Vectoriseur TF-IDF sauvegardé
└── 📁 notebooks/                           # Notebooks Jupyter
├    ├── 📄 01_data_analysis.ipynb          # Analyse et nettoyage
├    ├── 📄 02_preprocessing.ipynb          # Tokenisation et stemming
├    └── 📄 03_modeling.ipynb               # Entrainement et evaluation
```

---

## Cloner et installer le projet

Pour utiliser ce projet sur votre machine locale, suivez ces étapes :

1. **Cloner le dépôt GitHub :**

```bash
git clone https://github.com/anass17/Systeme-Anti-Spam-Intelligent
cd Systeme-Anti-Spam-Intelligent
```

2. **Créer un environnement virtuel (recommandé) :**

```bash
python -m venv venv
venv\Scripts\activate     # Sur Windows
source venv/bin/activate  # Sur Linux / Mac
```

3. **Installer les dépendances :**

```bash
pip install -r requirements.txt
```

4. **Lancer l’application Streamlit :**

```bash
streamlit run app/streamlit_app.py
```

5. **Ouvrir l’application dans votre navigateur:**
Streamlit ouvrira automatiquement une fenêtre locale, sinon rendez-vous sur : http://localhost:8501/


---

## 1. Prétraitement du texte

Les étapes suivantes ont été appliquées :

* Conversion en minuscules
* Suppression de la ponctuation et des caractères spéciaux (regex)
* Suppression des stopwords (NLTK)
* Tokenisation
* Stemming avec **PorterStemmer**
* Recomposition des tokens en texte nettoyé

Ces étapes garantissent des entrées cohérentes et significatives pour le modèle.

---

## 2. Extraction des caractéristiques (TF-IDF)

`TfidfVectorizer` a été utilisé pour transformer le texte en vecteurs numériques.

Paramètres clés :

* `max_features=5000` → conserver uniquement les mots les plus importants

La matrice TF-IDF a servi à l'entraînement des modèles.

---

## 3. Modèles de machine learning

Plusieurs modèles ont été entraînés et évalués :

* Régression Logistique
* Linear SVM
* Naive Bayes
* Random Forest
* SGDClassifier

### Optimisation des hyperparamètres

`GridSearchCV` a été utilisé pour trouver les meilleurs paramètres pour :

* Régression Logistique
* Linear SVM
* Naive Bayes
* SGDClassifier

---

## 4. Meilleur modèle

Après optimisation, le meilleur modèle est :

### **Linear SVM (Support Vector Machine)**

* Meilleur F1-score : **0.98841**
* Meilleure précision (accuracy) : **0.98837**

Ce modèle a été sauvegardé sous `spam_classifier_model.pkl`.

---

## 5. Sauvegarde du modèle

Nous avons sauvegardé :

* Le modèle ML (SVM) entraîné
* Le vectoriseur TF-IDF

Avec joblib :

```python
joblib.dump(best_model, "spam_classifier_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
```

---

## 6. Déploiement avec Streamlit

Le projet inclut `streamlit_app.py` qui :

1. Charge le modèle et le TF-IDF sauvegardés
2. Prend un texte en entrée de l'utilisateur
3. Prédit s'il s'agit d'un spam ou non

Exemple pour lancer l'application :

```bash
streamlit run app/streamlit_app.py
```

---

## 7. Dépendances

`requirements.txt` :

```
streamlit
scikit-learn
pandas
numpy
nltk
joblib
wordcloud
matplotlib
```

---

## 8. Résumé des résultats

| Modèle                | Meilleur F1  | Meilleure précision |
| --------------------- | ------------ | ------------------- |
| Régression Logistique | 0.988156     | 0.988196            |
| SVM                   | **0.988411** | 0.988370            |
| Naive Bayes           | 0.982420     | 0.981774            |
| SGD                   | 0.988022     | **0.988891**        |

**Choix final → SVM car il a le meilleur F1-score.**

---

## 9. Visualisations du projet

### Word Cloud des mots fréquents
![Spam Word Cloud](https://github.com/user-attachments/assets/7e64fe0e-80c0-4a83-a510-036aeec6ad01)
![Ham Word Cloud](https://github.com/user-attachments/assets/3d496b3a-8695-4b35-af6c-85bf186bf7ec)

### Interface Streamlit
![Streamlit UI](https://github.com/user-attachments/assets/a650e83a-b6cd-48f6-911d-e218ec37d07d)

---

## Conclusion

Ce projet démontre un pipeline complet de **machine learning** et de **NLP** pour la détection de spam, du prétraitement des textes au déploiement. Avec un **SVM** performant et une interface **Streamlit**, le système est prêt pour un usage réel.