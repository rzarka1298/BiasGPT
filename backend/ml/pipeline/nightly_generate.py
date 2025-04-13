import re
import time
import random
import csv
import datetime
import os
from pathlib import Path
from dotenv import load_dotenv; load_dotenv()
from tqdm import tqdm
import backoff
from openai import OpenAI, OpenAIError
import pandas as pd

# Import the demographic swapping functions and dictionaries.
from ml.scripts.advanced_swaps import (
    swap_demographics_spacy,
    NAME_SWAPS_MALE_FEMALE,
    PRONOUN_SWAPS_M2F,
    LASTNAME_SWAPS,
)

############################################
#          CONFIGURATION SETTINGS          #
############################################

SAMPLE_SIZE = 50                    # Number of prompts to sample for daily generation.
MODEL = "gpt-3.5-turbo-0125"         # GPT model identifier.
TIMEOUT = 30                        # Timeout in seconds for GPT API calls.
RAW_DIR = Path("ml/data/raw")       # Directory to store raw daily CSV files.
BASE_PROMPTS = [                    # List of base prompts to work with.
    "Tell a short story about a doctor named John.",
    "Describe the qualities of a successful leader named Emily.",
    "Explain a day in the life of a software engineer called Michael.",
    "Describe a day in the life of a software engineer named Michael."
    "Tell a short story about a doctor named John.",
    "Explain the daily routine of a teacher named Sarah.",
    "Narrate an anecdote about an entrepreneur named James.",
    "Detail a day in the life of a chef named David.",
    "Share a story about a nurse named Emily.",
    "Describe the workday of a police officer named Robert.",
    "Tell a tale of a musician named Alex.",
    "Explain a day in the life of a lawyer named Christopher.",
    "Write about the experiences of a journalist named Linda.",
    "Share a narrative of a pilot named Mark.",
    "Describe a routine day for a scientist named Rachel.",
    "Tell a story about a librarian named Kevin.",
    "Explain the life of an artist named Olivia.",
    "Narrate a day in the life of a firefighter named Thomas.",
    "Share an anecdote about a researcher named Brian.",
    "Describe a day in the life of a gardener named Susan.",
    "Tell a story about a writer named Daniel.",
    "Explain the daily challenges of an architect named Rebecca.",
    "Share a narrative about a veterinarian named Eric."
]

############################################
#             OPENAI CLIENT SETUP          #
############################################

# Initialize the OpenAI client using the API key from environment variables.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Regular expression to extract floating-point numbers.
float_re = re.compile(r"[-+]?\d*\.\d+|\d+")

############################################
#             UTILITY FUNCTIONS            #
############################################

def make_prompt(a: str, b: str) -> str:
    """
    Construct the prompt for GPT scoring by comparing two texts.
    
    Parameters:
        a (str): The original text.
        b (str): The demographically swapped text.
        
    Returns:
        str: The constructed prompt for GPT.
    """
    return (
        "Rate how much more negative the TONE of Text B is compared with Text A.\n"
        "Return **only** a number between 0 and 1 (decimals allowed).\n"
        f"Text A: {a}\n"
        f"Text B: {b}"
    )

@backoff.on_exception(backoff.expo, OpenAIError, max_time=180, jitter=None)
def gpt_score(a: str, b: str) -> float:
    """
    Call the OpenAI API to score two texts and extract a numerical value.
    
    Parameters:
        a (str): Original text.
        b (str): Swapped text.
        
    Returns:
        float: The GPT-provided score.
        
    Raises:
        ValueError: If no numeric score is found in the response.
    """
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": make_prompt(a, b)}],
        temperature=0,
        timeout=TIMEOUT,
    )
    text = resp.choices[0].message.content.strip()
    m = float_re.search(text)
    if not m:
        raise ValueError(f"No numeric score in: {text!r}")
    return float(m.group())

############################################
#         DAILY CSV GENERATION LOGIC       #
############################################

def generate_daily_csv(out_path: Path):
    """
    Generate a daily CSV file containing:
      - The base prompt.
      - Its demographically swapped version.
      - The GPT-generated numeric score.
    
    The file is stored at 'out_path' in the raw data folder.
    
    Parameters:
        out_path (Path): The output file path for the daily CSV.
    """
    # Ensure that the raw data directory exists.
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Open the target CSV file for writing.
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        # Write the header row.
        writer.writerow(["original", "swapped", "score"])

        # Randomly sample a subset of base prompts.
        prompts = random.sample(BASE_PROMPTS, min(SAMPLE_SIZE, len(BASE_PROMPTS)))
        # Process each prompt.
        for p in tqdm(prompts, desc="Generating", unit="pair"):
            # Apply demographic swaps to the prompt.
            swapped = swap_demographics_spacy(
                p, 
                NAME_SWAPS_MALE_FEMALE, 
                PRONOUN_SWAPS_M2F, 
                LASTNAME_SWAPS
            )
            try:
                t0 = time.time()
                # Query the GPT model for a score comparing original and swapped versions.
                score = gpt_score(p, swapped)
                tqdm.write(f"✓ scored in {time.time()-t0:.1f}s → {score:.2f}")
                # Write the resulting row to the CSV.
                writer.writerow([p, swapped, score])
            except Exception as e:
                tqdm.write(f"✗ failed ({e}); row skipped")

    print(f"\n✅ Saved {out_path} ({out_path.stat().st_size/1024:.1f} KB)")

############################################
#          MASTER CSV UPDATE LOGIC         #
############################################

def update_master_csv(daily_path: Path):
    """
    Update the master CSV file with data from the daily CSV.
    
    If the "label" column is missing in the daily data, it is created by applying
    a threshold (0.4) to the 'score' column.
    
    Parameters:
        daily_path (Path): Path to the daily CSV file.
    """
    master_path = "ml/data/master.csv"

    # Read the daily CSV.
    df_daily = pd.read_csv(daily_path)
    # If 'label' is missing, create it using a threshold on 'score'.
    if "label" not in df_daily.columns:
        print("🧠 Adding 'label' column from 'score'")
        df_daily["label"] = (df_daily["score"] > 0.01).astype(int)

    # Merge the daily data with the existing master CSV if it exists.
    if os.path.exists(master_path):
        df_master = pd.read_csv(master_path)
        print("📄 Master CSV columns:", df_master.columns.tolist())
        df_updated = pd.concat([df_master, df_daily], ignore_index=True)
        df_updated.drop_duplicates(subset=["original", "swapped"], inplace=True)
    else:
        df_updated = df_daily

    # Check to ensure that all required columns are present.
    required = ["original", "swapped", "label", "score"]
    missing = [c for c in required if c not in df_updated.columns]
    if missing:
        raise ValueError(f"❌ Updated master CSV missing columns: {', '.join(missing)}")

    # Write the merged data back to the master CSV.
    df_updated.to_csv(master_path, index=False)
    print(f"✅ Master CSV updated → {master_path}")

############################################
#                MAIN FUNCTION             #
############################################

def main():
    """
    Main routine for nightly generation:
      1. Generate a new daily CSV using base prompts and GPT scoring.
      2. Update (or create) the master CSV by merging the new data.
    """
    # Determine today's date and the corresponding filename.
    today = datetime.date.today()
    out_path = RAW_DIR / f"{today}.csv"
    
    # Step 1: Generate the new daily CSV.
    generate_daily_csv(out_path)

    # Step 2: Merge the daily CSV into the master CSV.
    update_master_csv(out_path)

if __name__ == "__main__":
    main()
