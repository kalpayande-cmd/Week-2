import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Step 1: Load Dataset
data = pd.read_csv("dataset/commands.csv")

# Step 2: Split into train/test
X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: TF-IDF Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1,3))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 4: Train Model
model = RandomForestClassifier()
model.fit(X_train_tfidf, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("✅ Model Training Complete!")
print(f"🎯 Accuracy: {accuracy*100:.2f}%")

# Step 6: Save Model and Vectorizer
with open("model/trained_model.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)

print("📦 Model saved successfully to 'model/trained_model.pkl'")
