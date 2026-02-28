
import os
from dotenv import load_dotenv
load_dotenv()

try:
    from google import genai
    print("✅ google-genai installed")
except ImportError:
    print("❌ google-genai NOT installed")
    exit(1)

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ GOOGLE_API_KEY not found in env")
    exit(1)

model = os.getenv("GEMINI_MODEL_NAME", "gemini-3-flash-preview")
print(f"Testing connectivity to model: {model}...")

try:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents="Say 'Connection Successful'"
    )
    print(f"Response: {response.text}")
    print("✅ Gemini API is WORKING")
except Exception as e:
    print(f"❌ Gemini API Failed: {e}")
    print("\nTroubleshooting:")
    print("1. Check if GOOGLE_API_KEY is correct")
    print("2. Check if model name represents a valid model you have access to")
    print("3. Try 'gemini-1.5-flash' if 'gemini-3-flash-preview' fails")
