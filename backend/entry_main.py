from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pydantic
from Notebooks import utils
import re
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, AutoTokenizer

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model=TFDistilBertForSequenceClassification.from_pretrained('./Models/chunk_model1')
tokenizer=utils.load_tokenizer('distilbert-base-uncased')

#Pydantic model for tweet input
class TwitterTweet(pydantic.BaseModel):
    text: str

#Function to clean the tweet text
def clean_text(text):
    text = re.sub(r'@\S+', '@user', text)
    text = re.sub(r'http\S+|www\S+', '', text)                           
    text = re.sub(r'#(\w+)', r'\1', text)                               
    text = re.sub(r'\s+', ' ', text).strip()                             
    return text

#API endpoint to classify the tweet sentiment
@app.post("/tweet")
async def post_tweet(tweet: TwitterTweet):
    
    cleaned_text = clean_text(tweet.text)
    inputs = tokenizer(cleaned_text, return_tensors='tf', padding='max_length', truncation=True, max_length=128)

    outputs = model.predict(inputs)
    logits = outputs.logits
    prob = tf.nn.sigmoid(logits)[0][0].numpy()
    sentiment = 'positive' if prob >= 0.5 else 'negative'
    return {"sentiment": sentiment, "confidence": float(prob)}