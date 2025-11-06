import json
from fastapi.testclient import TestClient
import main

client = TestClient(main.app)

def test_home():
    r = client.get("/")
    assert r.status_code == 200

def test_api():
    # Use a category that exists in your features
    cat = main.FE["category"].dropna().astype(str).unique()[0]
    payload = {"category": cat, "brand": None, "size_grams": None, "top_n": 3}
    r = client.post("/api/recommend", json=payload)
    assert r.status_code == 200
    assert isinstance(r.json(), list)
