import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

os.makedirs("model", exist_ok=True)

df = pd.read_csv("data.csv")
X = df["text"].astype(str)
y = df["label"]

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000)),
])

pipe.fit(X, y)

joblib.dump(pipe, "model/text_model.joblib")
print(f"Text model saved. Classes: {pipe.classes_}")
