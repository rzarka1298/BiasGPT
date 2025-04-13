import glob
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file.
load_dotenv()

# Define file paths and patterns.
RAW_DIR_PATTERN = "ml/data/raw/*.csv"  # Pattern to capture all raw CSV files
LABELED_DIR_PATTERN = "ml/data/labeled/*.csv"  # Pattern for labeled CSVs (if needed)
MASTER_PATH = "ml/data/master.csv"  # Path to the master CSV file

def load_dataset() -> pd.DataFrame:
    """
    Load the latest raw CSV from the RAW_DIR_PATTERN location.
    
    Ensures the CSV contains the required columns: 'original', 'swapped', 'label', and 'score'.  
    If the 'label' column is missing and 'score' exists, it creates the label column using a threshold of 0.5.
    
    Returns:
        pd.DataFrame: The loaded and processed DataFrame.
        
    Raises:
        FileNotFoundError: If no raw CSV files are found.
        ValueError: If a required column (other than label) is missing.
    """
    # Get a sorted list of raw CSV file paths.
    raw_files = sorted(glob.glob(RAW_DIR_PATTERN))
    if not raw_files:
        raise FileNotFoundError("No raw data files found in ml/data/raw/")

    # Use the latest file (last item in the sorted list).
    latest_file = raw_files[-1]
    print(f"🔄 Loading latest raw file: {latest_file}")
    df = pd.read_csv(latest_file)
    
    # Define the required columns.
    required_columns = ["original", "swapped", "label", "score"]
    # Ensure each required column is present.
    for col in required_columns:
        if col not in df.columns:
            if col == "label" and "score" in df.columns:
                print("🧠 Creating missing 'label' column from 'score' using threshold 0.5")
                df["label"] = (df["score"] > 0.01).astype(int)
            else:
                raise ValueError(f"Missing required column: {col}")
    
    return df

def load_raw() -> pd.DataFrame:
    """
    Load and return a DataFrame with only the required columns from the latest raw CSV.
    
    Returns:
        pd.DataFrame: DataFrame containing columns 'original', 'swapped', 'score', and 'label'
    """
    df = load_dataset()
    # Return only the specified columns.
    return df[["original", "swapped", "score", "label"]]

def load_master() -> pd.DataFrame:
    """
    Load the master CSV if it exists; otherwise, create an empty DataFrame with expected columns.
    
    Returns:
        pd.DataFrame: The master DataFrame.
    """
    if os.path.exists(MASTER_PATH):
        return pd.read_csv(MASTER_PATH)
    else:
        # Create an empty DataFrame with columns matching what we expect.
        return pd.DataFrame(columns=["original", "swapped", "label"])

def main():
    """
    Main routine to merge new raw data with the master CSV and flag uncertain examples.
    
    Steps:
      1. Load existing master data (if any).
      2. Load the latest raw data.
      3. Merge the two, dropping duplicate rows.
      4. Save the updated master CSV.
      5. Flag uncertain rows (those with label=0 and score between 0.3 and 0.7) for manual review.
    """
    # Load master and new data.
    master = load_master()
    new = load_raw()
    print(f"📊 Merging {len(new)} new rows with {len(master)} existing rows...")
    
    # Concatenate the master and new datasets; drop duplicate rows.
    merged = pd.concat([master, new]).drop_duplicates()
    merged.to_csv(MASTER_PATH, index=False)
    print(f"✅ Updated master CSV with {len(merged)} total rows.")

    # Identify "uncertain" examples for review.
    uncertain = new[(new["label"] == 0) & (new["score"].between(0.3, 0.7))]
    if not uncertain.empty:
        uncertain.to_csv("ml/data/to_label.csv", index=False)
        print(f"⚠️ {len(uncertain)} rows need manual review → ml/data/to_label.csv")
    else:
        print("✅ No uncertain rows found.")

if __name__ == "__main__":
    main()
