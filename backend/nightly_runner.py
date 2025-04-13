import schedule
import time
import subprocess
import datetime

def run_pipeline():
    """
    Runs your BiasGPT pipeline by executing the shell script.
    Logs the current time and prints a message before invoking the script.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] Running BiasGPT pipeline…")
    subprocess.run([
        "/bin/bash",
        "/Users/rugvedzarkar/Desktop/InnovateProject/backend/run_pipeline.sh"
    ])

# Schedule the pipeline to run every 30 minutes.
schedule.every(2).minutes.do(run_pipeline)

print("📅 BiasGPT runner started. Running every 2 minutes…")

while True:
    schedule.run_pending()
    time.sleep(60)  # Check pending tasks every minute.
