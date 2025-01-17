import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Chargement du modèle préentrainé
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def predict_spam(message):
    # Transformation du message
    message_vec = vectorizer.transform([message])
    
    # Prédiction
    prediction = model.predict(message_vec)
    
    # Retourner 'spam' ou 'ham' selon la prédiction
    return 'spam' if prediction[0] == 'Le message est un spam' else 'Le message est un ham'
