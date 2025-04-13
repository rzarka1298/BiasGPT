from openai import OpenAI
import os, time, dotenv; dotenv.load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

t0 = time.time()
print("Sending…")
r = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "ping"}],
    timeout=30        # ← new name
)
print("Done in %.1fs →" % (time.time() - t0), r.choices[0].message.content)
