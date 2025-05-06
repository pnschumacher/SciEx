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
COURSE_MATERIAL_TYPE=${6:-"slides"}
EMBEDDING_MODEL=${7:-"BAAI/bge-m3"}
EMBEDDING_MODEL_PATH=${8:-"$DEF_EMBEDDING_MODEL_PATH"}
SIMILARITY_TOP_K=${9:-"10"}
VECTOR_DB_PATH=${10:-"$DEF_VECTOR_DB_PATH"}  # defined in env_vars.sh
TRANSCRIPT_CHUNK_SIZE=${11:-"300"} 
RETRIEVAL_CONTENT_TYPE=${12:-"text"} 
CONTEXT_CONTENT_TYPE=${13:-"text"} 

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
    --course-material-type ${COURSE_MATERIAL_TYPE} \
    --embedding-model-name ${EMBEDDING_MODEL} \
    --embedding-model-path ${EMBEDDING_MODEL_PATH} \
    --similarity-top-k ${SIMILARITY_TOP_K} \
    --vector-db-path ${VECTOR_DB_PATH} \
    --transcript-chunk-size ${TRANSCRIPT_CHUNK_SIZE} \
    --retrieval-slide-type ${RETRIEVAL_CONTENT_TYPE} \
    --context-slide-type ${CONTEXT_CONTENT_TYPE} \
    --exam-json-path ${file} \
    --use-course-material "true"

  echo "---------------------------------------------------------"
done