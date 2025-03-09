import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

print("GLOBAL_LLM_SERVICE:", os.getenv("GLOBAL_LLM_SERVICE"))
print("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME:", os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"))
print("AZURE_OPENAI_ENDPOINT:", os.getenv("AZURE_OPENAI_ENDPOINT"))
print("AZURE_OPENAI_API_KEY:", os.getenv("AZURE_OPENAI_API_KEY"))
print("AZURE_OPENAI_EMBED_DEPLOYMENT_NAME:", os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT_NAME"))
print("GEOCODING_API_KEY:", os.getenv("GEOCODING_API_KEY"))
print("AI_SEARCH_KEY:", os.getenv("AI_SEARCH_KEY"))
print("AI_SEARCH_URL:", os.getenv("AI_SEARCH_URL"))
print("AZURE_TEXT_TO_IMAGE_DEPLOYMENT_NAME:", os.getenv("AZURE_TEXT_TO_IMAGE_DEPLOYMENT_NAME"))
print("AZURE_TEXT_TO_IMAGE_ENDPOINT:", os.getenv("AZURE_TEXT_TO_IMAGE_ENDPOINT"))
print("AZURE_TEXT_TO_IMAGE_API_KEY:", os.getenv("AZURE_TEXT_TO_IMAGE_API_KEY"))