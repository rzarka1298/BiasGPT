from dotenv import load_dotenv
import os

load_dotenv()                    # loads backend/.env by default
print("Key starts with:", os.getenv("OPENAI_API_KEY")[:4])
