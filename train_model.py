import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle


fake_df = pd.read_csv("Fake.csv")
real_df = pd.read_csv("True.csv")

fake_df["label"] = 0  # Fake
real_df["label"] = 1  # Real

data = pd.concat([fake_df, real_df])
data = data.sample(frac=1).reset_index(drop=True)

X = data['text']
y = data['label']


tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_tfidf = tfidf.fit_transform(X)


model = PassiveAggressiveClassifier(max_iter=100, class_weight="balanced")
model.fit(X_tfidf, y)



pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))

print("✅ Model trained and saved!")
