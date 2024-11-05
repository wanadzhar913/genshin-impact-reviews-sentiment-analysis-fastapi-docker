"""This is a script to create a Train-test splitted dataset using Spacy for lemmatization."""

import re
import warnings

import pandas as pd

import spacy
import fasttext
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

# Load spaCy model, https://spacy.io/usage/processing-pipelines#disabling
nlp = spacy.load("en_core_web_sm", enable = ['lemmatizer'])

def lemmatize_text(
        raw_text: str,
    ) -> str:
    """
    This is a function to convert raw text to a string of meaningful words.

    ### Arguments
    - `raw_text`: The input text to pre-process.

    ### Returns
    A pre-processed string.
    """
    # Remove HTML tags
    review_text = BeautifulSoup(raw_text).get_text()
    
    # Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)

    stopwords_list = [
        'character', 'characters', 'even', 'feel', 'fun', 'game', 'Genshin', 'genshin', 
        'good', 'great', 'like', 'lot', 'love', 'make', 'much', 'need', 'play', 'player', 
        'playing', 'played', 'really', 'still', 'story', 'take', 'want',
    ]
    
    # Convert words to lower case and split each word up
    words = letters_only.lower().split()
    
    # Searching through a set is faster than searching through a list 
    # Hence, we will convert stopwords to a set
    stops = set(stopwords.words('english'))
    
    # Adding on stopwords that were appearing frequently in both positive and negative reviews 
    stops.update(stopwords_list)
    
    # Remove stopwords
    meaningful_words = [w for w in words if w not in stops]

    # Use spaCy for lemmatization
    doc = nlp(" ".join(meaningful_words))
    lemmatized_words = [token.lemma_ for token in doc]
   
    # Join words back into one string, with a space in between each word
    return(" ".join(lemmatized_words))

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # load dataset
    reviews = pd.read_csv('../data/genshin_impact_reviews_v03112024.csv', parse_dates=['at','repliedAt'])

    # Defining the target variable using scores
    reviews['target'] = reviews['score'].map(lambda x: 1 if x < 4 else 0)

    print('Gonna start lemmatizing the texts!')

    # Pre-process the raw text
    reviews['content_lemma'] = reviews['content'].map(lambda x: lemmatize_text(raw_text=x))

    print('Done lemmatizing the texts!')

    # Find the number of meaningful words in each review
    reviews['content_clean_len'] = reviews['content_lemma'].str.split().map(len)

    # Drop these reviews that do not have any meaningful words
    reviews = reviews.drop(reviews[reviews['content_clean_len']==0].index)

    # Reindex the dataframe
    reviews.reset_index(drop=True, inplace=True)

    # Remove reviews that are non-English or gibberish using Fasttext
    model_path = '/home/faiq0913/.cache/huggingface/hub/models--facebook--fasttext-language-identification/snapshots/3af127d4124fc58b75666f3594bb5143b9757e78/model.bin'
    model = fasttext.load_model(model_path)

    reviews['language'] = reviews['content'].map(lambda x: model.predict(x)[0][0])

    reviews_cleaned = reviews.loc[reviews['language'].str.contains('eng')]

    # As we would like to stratify our target variable, we will need to first assign X and y
    X = reviews_cleaned[[cols for cols in reviews_cleaned.columns if cols != 'target']]
    y = reviews_cleaned['target']

    # Perform a train_test_split to create a train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Merge X_train and y_train back together using index
    train = pd.merge(X_train, y_train, left_index=True, right_index=True)

    # Merge X_test and y_test back together using index
    test = pd.merge(X_test, y_test, left_index=True, right_index=True)

    # Reindex the train and test set
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    print(f'Train shape: {train.shape}')
    print(f'Test shape: {test.shape}')

    # Keep only the columns that we need for modeling and interpretation
    train = train[['content','content_lemma','score','target']]
    test = test[['content','content_lemma','score','target']]

    # Save clean training set
    train.to_csv('../data/clean_train_lemma.csv', index=False)

    # Save clean test set
    test.to_csv('../data/clean_test_lemma.csv', index=False)
