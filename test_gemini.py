import os
from google import genai
from google.genai import types

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# Clean Config without AFC warnings
config = types.GenerateContentConfig(
    temperature=0.1,
    response_mime_type="application/json",
    tools=[],  
)

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Say hello in JSON: {\"message\": \"hello\"}",
    config=config
)

print("✅ Success:", response.text)