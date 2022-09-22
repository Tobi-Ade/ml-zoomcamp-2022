"""
Sending a request to the flask app
"""
import requests

url = "http://localhost:8080/pred"

customer = {"contract": "two_year", "tenure": 1, "monthlycharges": 10}

response = requests.post(url, json=customer).json()

print(response)

