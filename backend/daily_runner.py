import schedule
import time
import subprocess
import datetime

def run_pipeline():
    """
    Run the BiasGPT pipeline by executing the run_pipeline.sh shell script.
    
    This function:
      - Prints the current timestamp and a log message.
      - Uses subprocess.run() to call the shell script.
    """
    # Get the current timestamp in a human-readable format.
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] Running BiasGPT pipeline…")
    
    # Run the pipeline shell script.
    # Adjust the path as necessary to point to your run_pipeline.sh.
    subprocess.run([
        "/bin/bash",
        "/Users/rugvedzarkar/Desktop/InnovateProject/backend/run_pipeline.sh"
    ])

# ------------------------------------------------------------------------------
# Schedule Configuration
# ------------------------------------------------------------------------------

# Schedule the run_pipeline() function to execute daily at 2:00 AM.
schedule.every().day.at("02:00").do(run_pipeline)

print("📅 BiasGPT daily runner started. Waiting for 2:00 AM…")

# ------------------------------------------------------------------------------
# Main Loop
# ------------------------------------------------------------------------------

# Continuously check for pending scheduled tasks and run them.
while True:
    schedule.run_pending()
    # Sleep for 60 seconds to reduce CPU usage between checks.
    time.sleep(60)
