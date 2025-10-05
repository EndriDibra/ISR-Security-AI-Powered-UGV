import pandas as pd
import requests

def prepare_prompt(csv_path, question, max_rows=50):
    df = pd.read_csv(csv_path)
    df = df.tail(max_rows)  # limit to latest N rows
    csv_str = df.to_csv(index=False)

    prompt = f"""
You are a smart robot assistant analyzing UGV sensor data.
Each row contains: Timestamp, Temperature, Humidity, Gas Level, and Anomaly Flag.

CSV Data:
{csv_str}

Question: {question}
"""
    return prompt

def ask_gemma_about_csv(prompt):
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "gemma3",
        "prompt": prompt,
        "stream": False
    })
    return response.json()["response"]

# Example use
csv_path = "SensorData.csv"
question = "When was the last anomaly and what were the sensor readings?"
prompt = prepare_prompt(csv_path, question)
response = ask_gemma_about_csv(prompt)

print("\n🧠 Gemma's Response:")
print(response)
