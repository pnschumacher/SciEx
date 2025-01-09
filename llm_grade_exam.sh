#!/bin/bash
set -eu  # Crash if variable used without being set

source venv/bin/activate
source env_vars.sh
export LD_LIBRARY_PATH=$HOME/.local/lib:${LD_LIBRARY_PATH:-} # this is done so python uses correct sqlite3 version

SERVER_TYPE=${1:-"openai"}
LLM_NAME=${2:-"llama3.3"}
LLM_NAME_FULL=${3:-"meta-llama/Llama-3.3-70B-Instruct"}
SERVER_URL=${4:-"http://127.0.0.1:8080"}  # Local llama.cpp, needs to be deployed first on same node
NR_SHOT=${5:-1}
SHOT_TYPE=${6:-"same_question"}
REF=${7:-"yes"}

# Loop through each JSON file in the current directory and its subdirectories
for file in $(find exams_json/ -type f -name '*.json'); do
  echo "Processing exam at $file"
  echo "Sending grading request ..."
  python -u llm_grade_exam.py \
    --server-type ${SERVER_TYPE} \
    --server-url ${SERVER_URL} \
    --llm-name-full ${LLM_NAME_FULL} \
    --llm-name ${LLM_NAME} \
    --exam-json-path ${file} \
    --nr-shots ${NR_SHOT} \
    --shot-type ${SHOT_TYPE} \
    --with-ref ${REF}

  echo "---------------------------------------------------------"
done
