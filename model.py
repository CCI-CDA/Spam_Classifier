import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Charger le dataset
df = pd.read_csv('data/SMSSpamCollection.txt', sep='\t', header=None, names=['label', 'message'])

# Prétraiter les données
X = df['message']
y = df['label']

# Diviser en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorisation
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)

# Entraîner le modèle
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Sauvegarder le modèle et le vectoriseur
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
