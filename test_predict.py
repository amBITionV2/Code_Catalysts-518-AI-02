import requests
import json

url = "http://localhost:5000/predict"
data = {
    "rank": 42000,
    "category": "GM",
    "location": "BANGALORE",
    "branch": "CSE",
    "topn": 10
}

response = requests.post(url, json=data)
print(f"Status Code: {response.status_code}")
print(f"Response:\n{json.dumps(response.json(), indent=2)}")
