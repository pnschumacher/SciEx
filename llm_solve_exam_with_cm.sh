#!/bin/bash
set -eu  # Crash if variable used without being set

source venv/bin/activate
source env_vars.sh
export LD_LIBRARY_PATH=$HOME/.local/lib:${LD_LIBRARY_PATH:-} # this is done so python uses correct sqlite3 version

SERVER_TYPE=${1:-"openai"}
LLM_NAME=${2:-"llama3.3"}
LLM_NAME_FULL=${3:-"meta-llama/Llama-3.3-70B-Instruct"}
SERVER_URL=${4:-"http://127.0.0.1:8080"}  # Local llama.cpp, needs to be deployed first on same node
COURSE_MATERIAL_PATH=${5:-"$DEF_COURSE_MATERIAL_PATH"}  # defined in env_vars.sh
EMBEDDING_MODEL=${6:-"BAAI/bge-m3"}
SIMILARITY_TOP_K=${7:-"10"}

# Loop through each JSON file in the current directory and its subdirectories
for file in $(find exams_json/ -type f -name '*.json'); do
  echo "Processing exam at $file"
  echo "Format checking ... "
  python -u validate_exam_json.py \
    --json_path ${file}

  echo "Sending request ..."
  python -u llm_solve_exam.py \
    --server-type ${SERVER_TYPE} \
    --server-url ${SERVER_URL} \
    --llm-name-full ${LLM_NAME_FULL} \
    --llm-name ${LLM_NAME} \
    --course-material-path ${COURSE_MATERIAL_PATH} \
    --embedding-model-name ${EMBEDDING_MODEL} \
    --similarity-top-k ${SIMILARITY_TOP_K} \
    --exam-json-path ${file} \
    --use-course-material "true"

  echo "---------------------------------------------------------"
done