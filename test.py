import requests

response = requests.post("http://127.0.0.1:5100/api/recommendations", json={
    "user_id": "user_90",
    "top_n": 5
})

data = response.json()
for rec in data["recommendations"]:
    print(f"{rec['providerName']} — score: {rec['hybrid_score']:.2f}")