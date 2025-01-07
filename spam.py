import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Charger les données
data = pd.read_csv("data/SMSSpamCollection.txt", sep="\t", header=None, names=["label", "message"])

# Vérifier le contenu
print(data.head())
print(data["label"].value_counts())

data['label'] = data['label'].map({'ham': 0, 'spam': 1})


def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Supprimer les caractères spéciaux
    text = text.lower().strip()
    return text

data['message'] = data['message'].apply(preprocess_text)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)



vectorizer = TfidfVectorizer(max_features=5000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)



model = MultinomialNB()
model.fit(X_train_vect, y_train)

# Évaluer le modèle
y_pred = model.predict(X_test_vect)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))



with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
