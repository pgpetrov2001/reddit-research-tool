import json
import requests
import time

URL = "http://localhost:8080"   # server address

def send_data():
    payload = {
        "device": "sensor-01",
        "temp": 22.5,
        "status": "OK"
    }

    print("Sending JSON:", payload)

    response = requests.post(
        URL,
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    print("Server response:", response.text)


if __name__ == "__main__":
    while True:
        send_data()
        time.sleep(5)   # send every 5 seconds

