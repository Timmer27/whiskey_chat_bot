from openai import OpenAI
import os

def get_llm_client():
    api_key = os.getenv("GROQ_API_KEY")
    base_url = os.getenv("GROQ_BASE_URL")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing")
    
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    return client