from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from fastapi import Request

app = FastAPI()

templates = Jinja2Templates(directory="templates")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

data = pd.read_csv("data/SMSSpamCollection.txt", sep="\t", header=None, names=["label", "message"])
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower().strip()
    return text

data['message'] = data['message'].apply(preprocess_text)

def predict_message(message):
    vect_message = vectorizer.transform([message])
    prediction = model.predict(vect_message)[0]
    return "spam" if prediction == 1 else "ham"

@app.get("/", response_class=HTMLResponse)
async def display_messages(request: Request):
    data['prediction'] = data['message'].apply(predict_message)
    
    return templates.TemplateResponse("messages.html", {"request": request, "messages": data})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
