This README is in English. For the French version, see [README.md](README.md).

# Intelligent Anti-Spam System

A complete Machine Learning and NLP (Natural Language Processing) project for classifying emails as **spam** or **ham**, using text preprocessing, TF-IDF vectorization, and multiple classification algorithms. The final model is deployed with **Streamlit**.

---

## Project Overview

This project builds an intelligent system capable of detecting spam emails with high accuracy. It includes:

* Dataset exploration and cleaning
* Text preprocessing (tokenization, stopwords removal, stemming…)
* Feature extraction with **TF-IDF**
* Training and optimization of multiple ML models
* Saving the best-performing model
* Deployment through **Streamlit**

---

## Project Structure

```
📁 Intelligent-Anti-Spam-System
│
├── 📄 requirements.txt                     # Dependencies
├── 📄 README.md                            # Project documentation
│
├── 📁 app/
│   └── 📄 streamlit_app.py                 # Streamlit interface
│
├── 📁 data/
│   ├── 📁 raw/                             # Raw dataset
│   └── 📁 processed/                       # Cleaned dataset
│
├── 📁 models/
│   ├── 📄 spam_classifier_model.pkl        # Saved ML model (SVM)
│   └── 📄 tfidf_vectorizer.pkl             # Saved TF-IDF vectorizer
│
└── 📁 notebooks/
    ├── 📄 01_data_analysis.ipynb           # Analysis & cleaning
    ├── 📄 02_preprocessing.ipynb           # Text preprocessing
    └── 📄 03_modeling.ipynb                # Model training & evaluation
```

---

## Installation & Setup

1. **Clone the repository**

```bash
git clone https://github.com/anass17/Systeme-Anti-Spam-Intelligent
cd Systeme-Anti-Spam-Intelligent
```
2. **Create a virtual environment (recommended)**

```bash
python -m venv venv
venv\Scripts\activate     # Sur Windows
source venv/bin/activate  # Sur Linux / Mac
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Launch the Streamlit app**

```bash
streamlit run app/streamlit_app.py
```

5. Open the application
Streamlit opens automatically, otherwise visit: `http://localhost:8501/`

---

## 1. Text Preprocessing

The following steps were applied:

* Lowercasing
* Removing punctuation and special characters (regex)
* Stopwords removal (NLTK)
* Tokenization
* Stemming using **PorterStemmer**
* Reconstructing cleaned tokens into text

These steps ensure consistent and meaningful inputs for the ML models.

---

## 2. Feature Extraction (TF-IDF)

`TfidfVectorizer` was used to transform text into numeric vectors.

Key parameter:

* `max_features=5000` → keeps the most relevant words

The TF-IDF matrix is used to train all ML models.

---

## 3. Machine Learning Models

Several models were trained and evaluated:

* Logistic Regression
* Linear SVM
* Naive Bayes
* Random Forest
* SGDClassifier

### Hyperparameter Optimization

`GridSearchCV` was used to tune parameters for:

* Logistic Regression
* Linear SVM
* Naive Bayes
* SGDClassifier

--- 

## 4. Best Model

After tuning, the best-performing model is:

### **Linear SVM (Support Vector Machine)**

* Best F1-score: **0.98841**
* Best Accuracy: **0.98837**

Saved as: `spam_classifier_model.pkl`

---

## 5. Model Saving

We saved:

* The trained SVM model
* The TF-IDF vectorizer

Using joblib:

```python
joblib.dump(best_model, "spam_classifier_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
```

---

## 6. Deployment with Streamlit

The `streamlit_app.py`:

1. Loads the saved model and vectorizer
2. Accepts user input text
3. Predicts whether the email is spam or ham

Run it:

```bash
streamlit run app/streamlit_app.py
```

---

## 7. Dependencies

`requirements.txt`:

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

## 8. Results Summary
Model	Best F1-score	Best Accuracy
| Model                 | Best F1-score | Best Accuracy       |
| --------------------- | ------------- | ------------------- |
| Régression Logistique | 0.988156      | 0.988196            |
| SVM                   | **0.988411**  | 0.988370            |
| Naive Bayes           | 0.982420      | 0.981774            |
| SGD                   | 0.988022      | **0.988891**        |

**Final choice → SVM for the best F1-score.**

---

## 9. Project Visualizations

### Word Cloud of frequent spam words
![Spam Word Cloud](https://github.com/user-attachments/assets/7e64fe0e-80c0-4a83-a510-036aeec6ad01)
![Ham Word Cloud](https://github.com/user-attachments/assets/3d496b3a-8695-4b35-af6c-85bf186bf7ec)

### Streamlit Interface
![Streamlit UI](https://github.com/user-attachments/assets/a650e83a-b6cd-48f6-911d-e218ec37d07d)

---

## Conclusion

This project demonstrates a complete **machine learning** and **NLP** pipeline for spam detection — from text preprocessing and model training to deployment. With a high-performance **SVM** model and an intuitive **Streamlit** interface, the system is ready for real-world usage.