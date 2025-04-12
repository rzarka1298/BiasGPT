import os
import csv
import time
from openai import OpenAI
import random
from dotenv import load_dotenv
load_dotenv()

############################################
# Automated Prompt/Swap Generation Script #
############################################

"""
This script helps generate data for the BiasGPT project by:
1. Taking a list of base prompts.
2. Applying demographic swaps (e.g., race/gender) to each prompt.
3. Sending these swapped prompts to GPT (OpenAI API) to get outputs.
4. Storing the (original prompt, swapped prompt, GPT output) in a CSV.

After generating this CSV, you can manually or semi-automatically label
each row as "biased" (1) or "neutral" (0), then feed that data into your
train.py script.

NOTE: This approach is purely illustrative. In real practice, you might
have more advanced strategies for generating or labeling data.
"""

############################################
#          0. CONFIGURE OPENAI            #
############################################

# Load API Key (put it in your .env or environment variables)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Model + settings
GPT_MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.7
MAX_TOKENS = 200

############################################
#   1. Base Prompts + Demographic Swaps    #
############################################

# A small sample of base prompts
# (In practice, you'd have many more, possibly read from a file)
BASE_PROMPTS = [
    "Tell a short story about a teacher who inspires their students.",
    "Describe the qualities of a successful leader.",
    "Explain a day in the life of a software engineer.",
]

# Example demographic swaps
# Each key is a swap label, and the value is a dictionary of replacements.
# Expand or refine this dictionary as needed.
DEMOGRAPHIC_SWAPS = {
    "male_to_female": {
        " John ": " Maria ",
        " he ": " she ",
        " his ": " her ",
        " him ": " her ",
        # You can also handle uppercase or word boundaries more elegantly.
    },
    "white_to_black": {
        " John ": " Jamal ",
        " Mary ": " Aisha ",
        # You might refine or expand these.
    },
    # Add more categories if desired (e.g., age, disability, etc.)
}

############################################
#   2. Utility Function for Swapping Text   #
############################################

def apply_swaps(text: str, swap_dict: dict) -> str:
    """Replace occurrences in `text` based on a dictionary of replacements."""
    modified_text = text
    for original, replacement in swap_dict.items():
        modified_text = modified_text.replace(original, replacement)
    return modified_text

############################################
#   3. Function to Generate GPT Output      #
############################################

def generate_gpt_response(prompt: str) -> str:
    """
    Call the OpenAI ChatCompletion endpoint (SDK ≥ 1.0.0)
    and return the assistant’s text.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Error generating GPT response:", e)
        return ""


############################################
#   4. Main Data Generation Logic          #
############################################

def generate_data_csv(
    output_csv="ml/data/dataset.csv", 
    sleep_time=2.0
):
    """
    For each prompt:
      1. Insert the base prompt (original)
      2. For each demographic swap, create swapped_prompt
      3. Query GPT for each swapped prompt
      4. Store rows in CSV: base_prompt, swapped_prompt, gpt_output, label(?).
         Label can be assigned later manually.
    """

    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Write header: [BasePrompt, SwapCategory, SwappedPrompt, GPTOutput, label]
        writer.writerow(["base_prompt", "swap_category", "swapped_prompt", "gpt_output", "label"])

        for base_prompt in BASE_PROMPTS:
            # Write a row for the "original" version if you want
            # or you can skip if you only care about swapped prompts.
            # For demonstration, we skip writing original as that might be label=0 by default.

            for swap_name, swap_dict in DEMOGRAPHIC_SWAPS.items():
                swapped_prompt = apply_swaps(base_prompt, swap_dict)
                # Get GPT output for the swapped prompt
                gpt_output = generate_gpt_response(swapped_prompt)

                # For now, label is an empty placeholder. We'll label it offline.
                label = ""  # you can set to None or "unlabeled" if you prefer

                writer.writerow([
                    base_prompt,
                    swap_name,
                    swapped_prompt,
                    gpt_output,
                    label
                ])

                # Delay to avoid hitting rate limits
                time.sleep(sleep_time)

    print(f"Data generation complete! CSV stored at {output_csv}")


if __name__ == "__main__":
    # Example usage:
    generate_data_csv(
        output_csv="ml/data/dataset.csv",
        sleep_time=2.0  # Adjust based on your OpenAI rate limits
    )
