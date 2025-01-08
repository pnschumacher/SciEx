#!/bin/bash
set -eu  # Crash if variable used without being set

source venv/bin/activate
source env_vars.sh
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH # this is done so python uses correct sqlite3 version

SERVER_TYPE=$1
LLM_NAME=$2
LLM_NAME_FULL=$3
SERVER_URL=$4
COURSE_MATERIAL_PATH=$5
EMBEDDING_MODEL=$6
SIMILARITY_TOP_K=$7

if [ -z ${SERVER_TYPE} ]; then
  SERVER_TYPE="openai"
fi
if [ -z ${LLM_NAME} ]; then
  LLM_NAME="llama3.3"
fi
if [ -z ${LLM_NAME_FULL} ]; then
  LLM_NAME_FULL="meta-llama/Llama-3.3-70B-Instruct"
fi
if [ -z ${SERVER_URL} ]; then
  SERVER_URL="http://127.0.0.1:8080"  # Local llama.cpp, needs to be deployed first on same node
fi
if [ -z ${COURSE_MATERIAL_PATH} ]; then
  COURSE_MATERIAL_PATH="$DEF_COURSE_MATERIAL_PATH" # defined in env_vars.sh
fi
if [ -z ${EMBEDDING_MODEL} ]; then
  EMBEDDING_MODEL="BAAI/bge-large-en"
fi
if [ -z ${SIMILARITY_TOP_K} ]; then
  SIMILARITY_TOP_K="10"
fi

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
    --use-course_material true

  echo "---------------------------------------------------------"
done