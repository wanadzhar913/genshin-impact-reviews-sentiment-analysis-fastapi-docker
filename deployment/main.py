import re
import pickle
import logging
from typing import Dict
from sys import stdout

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from mangum import Mangum
from fastapi import FastAPI, APIRouter, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
from model import SentimentRNN

# Set up logging
logger = logging.getLogger()

logger.setLevel(logging.INFO)
logFormatter = logging.Formatter\
("%(name)-12s %(asctime)s %(levelname)-8s %(filename)s:%(funcName)s %(message)s")
consoleHandler = logging.StreamHandler(stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

# Load the vocab dictionary from the pickle file
with open("pytorch_vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# Load the trained LSTM model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SentimentRNN(
    no_layers=2,
    vocab_size=1502,
    output_dim=1,
    hidden_dim=256,
    embedding_dim=64,
    drop_prob=0.6,
    device=device,
)

model.load_state_dict(
    torch.load("rnn_pytorch_stem_corpus_v20241119.pth",
    map_location=device,
    weights_only=True,
))
model.to(device)
model.eval()

# Define a request schema
class SentimentRequest(BaseModel):
    text: str

# Text preprocessing function
def preprocess_text(raw_text: str) -> Dict:
    """
    Cleans and stems text, removes stopwords, and returns processed data. In general it:

    1. Remove HTML tags using BeautifulSoup.
    2. Keep only letters, replacing non-letter characters with spaces.
    3. Convert text to lowercase and split into words.
    4. Remove stopwords to keep only meaningful words.
    5. Apply stemming to reduce words to their root forms.
    6. Rejoin stemmed words into a processed text string.

    Returns:
        - original text
        - stemmed text
        - number of meaningful words
    """
    # Remove HTML tags
    review_text = BeautifulSoup(raw_text, "html.parser").get_text()

    # Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)

    # Convert to lowercase and split
    words = letters_only.lower().split()

    # Remove stopwords
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if w not in stops]

    # Apply stemming
    p_stemmer = PorterStemmer()
    stemmed_words = [p_stemmer.stem(w) for w in meaningful_words]
    stemmed_text = " ".join(stemmed_words)

    logger.info(f"Len of stemmed text: {len(stemmed_words)}")

    return {
        "original_text": raw_text,
        "stemmed_text": stemmed_text,
        "num_meaningful_words": len(stemmed_words),
    }

# Function to convert text into numerical format for model
def text_to_tensor(stemmed_text: str):
    tokens = stemmed_text.split()
    numerical_tokens = [vocab.get(token, vocab["<unk>"]) for token in tokens]
    input_tensor = torch.tensor([numerical_tokens], dtype=torch.long).to(device)
    return input_tensor

# Initialize FastAPI
app = FastAPI()

app = FastAPI(
    title="Faiq's Genshin Impact Google Review Sentiment Analyzer",
    summary="Endpoints for classifying Genshin Impact reviews into positive/negative sentiment using PyTorch LSTMs.",
    docs_url="/lstmparse/docs",
    openapi_url="/lstmparse/openapi.json",
)


router = APIRouter(prefix="/lstmparse")

@router.get(
    "/",
    status_code=status.HTTP_200_OK,
    tags=["Testing"],
)
async def root():
    return {"message": "This is the API's root!"}

@router.post(
    "/stem_text",
    status_code=status.HTTP_200_OK,
    tags=["Testing"],
)
async def display_pdf_text(text: str) -> Dict:
    """
    Preprocesses a given text by removing stopwords, stemming words, 
    and extracting meaningful words.

    This endpoint takes a raw text input, applies text preprocessing, 
    and returns the cleaned text along with other useful metadata.

    Args:
        text (str): The input text to be processed.

    Returns:
        dict: A dictionary containing:
            - "original_text" (str): The input text before preprocessing.
            - "stemmed_text" (str): The processed text after stemming and stopword removal.
            - "num_meaningful_words" (int): The number of meaningful words remaining after preprocessing.

    Example:
        Request:
        ```json
        {
            "text": "This is an amazing game! I highly recommend it."
        }
        ```

        Response:
        ```json
        {
            "original_text": "This is an amazing game! I highly recommend it.",
            "stemmed_text": "amaz game highli recommend",
            "num_meaningful_words": 4
        }
        ```
    """
    text_dict = preprocess_text(raw_text=text)

    return text_dict

# Prediction endpoint
@router.post(
    "/classify",
    status_code=status.HTTP_200_OK,
    tags=["Classify Texts"]
)
async def predict_sentiment(request: SentimentRequest):
    """
    Classifies the sentiment of a given review as either positive or negative.

    This endpoint takes in a text input, preprocesses it by stemming and 
    removing stopwords, converts it into a tensor, and feeds it into an 
    LSTM-based sentiment analysis model to determine its sentiment.

    Args:
        request (SentimentRequest): A request object containing the text to be classified ("text").

    Returns:
        dict: A dictionary containing:
            - "sentiment" (str): "positive" or "negative" based on the model's prediction.
            - "original_text" (str): The input text before preprocessing.
            - "stemmed_text" (str): The processed text after stemming and stopword removal.
            - "num_meaningful_words" (int): The number of meaningful words remaining after preprocessing.

    Example:
        Request:
        ```json
        {
            "text": "I absolutely love this game! It was fantastic!"
        }
        ```

        Response:
        ```json
        {
            "sentiment": "positive",
            "original_text": "I absolutely love this game! It was fantastic!",
            "stemmed_text": "absolut love game fantast",
            "num_meaningful_words": 4
        }
        ```
    """
    # Preprocess text
    processed = preprocess_text(request.text)

    # Convert to tensor
    input_tensor = text_to_tensor(processed["stemmed_text"])

    # Perform inference
    with torch.no_grad():
        output, _ = model(input_tensor)
        sentiment = int(torch.round(output.squeeze()).cpu().numpy())

    # Return response
    return {
        "sentiment": "positive" if sentiment == 0 else "negative",
        "original_text": processed["original_text"],
        "stemmed_text": processed["stemmed_text"],
        "num_meaningful_words": processed["num_meaningful_words"],
    }

app.add_middleware(
    CORSMiddleware,
    # allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
lambda_handler = Mangum(app)