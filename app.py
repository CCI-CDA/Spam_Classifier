from fastapi import FastAPI
from pydantic import BaseModel
from classifier import predict_spam

app = FastAPI()

class Message(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"message": "Spam Classifier API is running."}

@app.get("/check")
async def check(message: str):
    return {"resp": predict_spam(message)}
