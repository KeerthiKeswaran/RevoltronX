import requests

url = "http://127.0.0.1:8000/predict"  # Change the URL if your FastAPI app is running on a different address or port

# Example input text
article_input = {"article_text": "This is an example article for testing."}

# Make a POST request to the /predict endpoint
response = requests.post(url, json=article_input)

# Print the response
print("Response status code:", response.status_code)
print("Response JSON:", response.json())
