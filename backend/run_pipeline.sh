#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

############################################
#        Set up the Conda Environment      #
############################################

# Get the base directory of your Conda installation.
CONDA_BASE="$(conda info --base)"
echo "📍 Sourcing conda from $CONDA_BASE..."
# Source conda.sh to enable conda command functionality.
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate the desired conda environment (named "innovate").
echo "📍 Activating conda env: innovate"
conda activate innovate

############################################
#         Set the Project Environment      #
############################################

# Define the root directory of your project.
PROJECT="/Users/rugvedzarkar/Desktop/InnovateProject/backend"
# Update PYTHONPATH to include the project directory (ensuring modules are found).
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

# Change directory to the project root.
echo "📍 Changing directory to $PROJECT"
cd "$PROJECT" || exit 1

############################################
#              Logging Setup               #
############################################

# Define the log file path where the pipeline output will be written.
LOG="/Users/rugvedzarkar/cron_biasgpt.log"

############################################
#          Execute the Pipeline Steps      #
############################################

# Step 1: Run the nightly_generate script to produce a new raw CSV.
echo "📍 Starting generate…" >> "$LOG"
# The output and errors are appended to the log file.
python -m ml.pipeline.nightly_generate >> "$LOG" 2>&1 || echo "generate step failed" >> "$LOG"

# Step 2: Run merge_and_split to update the master CSV.
echo "📍 Starting merge+split…" >> "$LOG"
python -m ml.pipeline.merge_and_split >> "$LOG" 2>&1 || echo "merge step failed" >> "$LOG"

# Step 3: Run train_and_deploy to train a new model and deploy it if it is better.
echo "📍 Starting train+deploy…" >> "$LOG"
python -m ml.pipeline.train_and_deploy >> "$LOG" 2>&1 || echo "train step failed" >> "$LOG"

############################################
#         Final Log Message                #
############################################

echo "📍 DONE ✅ $(date)" >> "$LOG"
