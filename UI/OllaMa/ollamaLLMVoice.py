# Author: Endri Dibra
# Bachelor Thesis: Smart Unmanned Ground Vehicle
# Application: AI Voice Assistant using Gemma3 via Ollama

# Importing all required packages
import pandas as pd
import requests
import speech_recognition as sr
import pyttsx3
import os

# Initializing the speech engine for response output
engine = pyttsx3.init()
engine.setProperty('rate', 175)

# Defining CSV file path
csv_path = "SensorData.csv"

# Function to prepare prompt with recent CSV data
def prepare_prompt(csv_path, question, max_rows=50):
    try:
        df = pd.read_csv(csv_path)
        df = df.tail(max_rows)  # Get last N rows
        csv_str = df.to_csv(index=False)

        prompt = f"""
You are an AI assistant for a smart unmanned ground vehicle.
Below is recent sensor data from the robot:
Each row contains: Timestamp, Temperature, Humidity, Gas Level, Anomaly Flag.

CSV Data:
{csv_str}

Question: {question}
"""
        return prompt
    except Exception as e:
        return f"Error reading CSV: {e}"

# Function to send prompt to Gemma3 model via Ollama
def ask_gemma(prompt):
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "gemma3",
            "prompt": prompt,
            "stream": False
        })
        return response.json()["response"]
    except Exception as e:
        return f"Error connecting to Gemma: {e}"

# Function to speak text aloud
def speak(text):
    print(f"\n🧠 Gemma's Answer: {text}\n")
    engine.say(text)
    engine.runAndWait()

# Function to listen to a voice question
def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙️ Ask your question about UGV sensor data:")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        try:
            question = recognizer.recognize_google(audio)
            print(f"🗣️ You asked: {question}")
            return question
        except sr.UnknownValueError:
            print("Sorry, could not understand.")
            return None
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return None

# Main loop
if __name__ == "__main__":
    while True:
        question = listen()
        if question:
            prompt = prepare_prompt(csv_path, question)
            answer = ask_gemma(prompt)
            speak(answer)

        # Optional break
        print("\nPress [Enter] to ask again or type 'q' to quit.")
        if input().lower() == 'q':
            break
