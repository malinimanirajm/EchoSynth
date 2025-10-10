# feature_engineering.py

import pandas as pd
import re
from textblob import TextBlob
from nltk import word_tokenize, ngrams
from collections import Counter
import spacy
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

# -----------------------------
# Step 0: Download necessary NLTK resources
# -----------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

# -----------------------------
# Step 1: Load dataset
# -----------------------------
df = pd.read_csv("data/amazon_beauty_train.csv")

# Ensure content_clean exists
if 'content_clean' not in df.columns:
    df['content_clean'] = df['content'].apply(lambda x: x if isinstance(x, str) else '')

# -----------------------------
# Step 2: Basic Text Features
# -----------------------------
df['word_count'] = df['content_clean'].apply(lambda x: len(x.split()))
df['char_count'] = df['content_clean'].apply(len)
df['avg_word_len'] = df['content_clean'].apply(lambda x: sum(len(w) for w in x.split())/len(x.split()) if len(x.split())>0 else 0)
df['exclamation_count'] = df['content_clean'].apply(lambda x: x.count('!'))
df['question_count'] = df['content_clean'].apply(lambda x: x.count('?'))

# Sentiment polarity (TextBlob)
df['polarity'] = df['content_clean'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['subjectivity'] = df['content_clean'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# Sentiment using Vader
df['vader_pos'] = df['content_clean'].apply(lambda x: sia.polarity_scores(x)['pos'])
df['vader_neg'] = df['content_clean'].apply(lambda x: sia.polarity_scores(x)['neg'])
df['vader_neu'] = df['content_clean'].apply(lambda x: sia.polarity_scores(x)['neu'])
df['vader_compound'] = df['content_clean'].apply(lambda x: sia.polarity_scores(x)['compound'])

# -----------------------------
# Step 3: N-gram Features (top 5 bigrams)
# -----------------------------
def top_ngrams(texts, n=2, top_k=5):
    all_ngrams = []
    for text in texts:
        tokens = [w.lower() for w in word_tokenize(text) if w.isalpha() and w not in stop_words]
        all_ngrams += list(ngrams(tokens, n))
    counter = Counter(all_ngrams)
    return counter.most_common(top_k)

df['top_bigrams'] = [top_ngrams([x], n=2, top_k=5) for x in df['content_clean']]

# -----------------------------
# Step 4: Named Entity Recognition (NER)
# -----------------------------
nlp = spacy.load('en_core_web_sm')  # Make sure you have this installed: python -m spacy download en_core_web_sm

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

df['entities'] = df['content_clean'].apply(extract_entities)

# -----------------------------
# Step 5: Topic Modeling (BERTopic)
# -----------------------------
# NOTE: For large datasets, consider sampling
sample_texts = df['content_clean'].sample(5000, random_state=42).tolist()
topic_model = BERTopic(language="english")
topics, probs = topic_model.fit_transform(sample_texts)
df.loc[df.index[:5000], 'topic'] = topics

# -----------------------------
# Step 6: Embeddings for Vector DB
# -----------------------------
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
df['embedding'] = df['content_clean'].apply(lambda x: embed_model.encode(x).tolist())

# -----------------------------
# Step 7: Save engineered features
# -----------------------------
df.to_csv("data/amazon_beauty_features.csv", index=False)
print("Feature engineering completed. Saved to 'data/amazon_beauty_features.csv'.")
