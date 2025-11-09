import pickle

# Load the saved vectorizer and model
with open("model/trained_model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)   # match the save order!

# Get text input from user
command = input("Enter voice command text: ")

# Transform input text using the saved TF-IDF vectorizer
x = vectorizer.transform([command])

# Predict using the trained model
prediction = model.predict(x)

print(f"🧠 Predicted Command: {prediction[0]}")
