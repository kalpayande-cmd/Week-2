import speech_recognition as sr
import pickle

# Load model and vectorizer
with open("model/trained_model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

# Function to recognize speech
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙️ Say an aerospace command:")
        audio = recognizer.listen(source)
    
    try:
        text = recognizer.recognize_google(audio)
        print("🗣️ You said:", text)
        return text
    except:
        print("⚠️ Could not recognize speech.")
        return ""

# Main
if __name__ == "__main__":
    command = recognize_speech()
    if command:
        vectorized = vectorizer.transform([command])
        prediction = model.predict(vectorized)
        print("🧭 Predicted Command:", prediction[0])
