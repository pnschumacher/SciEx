#!/bin/bash
set -eu  # Crash if variable used without being set

# Setting environment
# source /home/tdinh/.bashrc
# conda activate py39
# . .llava/bin/activate
# which python

source venv/bin/activate


SERVER_TYPE=$1
LLM_NAME=$2
LLM_NAME_FULL=$3
SERVER_URL=$4

if [ -z ${SERVER_TYPE} ]; then
  SERVER_TYPE="openai"
fi
if [ -z ${LLM_NAME} ]; then
  LLM_NAME="mixtral"
fi
if [ -z ${LLM_NAME_FULL} ]; then
  LLM_NAME_FULL=mistralai/Mixtral-8x7B-Instruct-v0.1
fi
if [ -z ${SERVER_URL} ]; then
  SERVER_URL="http://i13hpc65:8080"  # Local mixtral lamma.cpp
fi
if [ -z ${USE_COURSE_MATERIAL} ]; then
  USE_COURSE_MATERIAL=0 
fi
if [ -z ${COURSE_MATERIAL_PATH} ]; then
  COURSE_MATERIAL_PATH="" 
fi
if [ -z ${EMBEDDING_MODEL} ]; then
  EMBEDDING_MODEL="BAAI/bge-large-en"
fi
if [ -z ${COURSE_MATERIAL_DB_PATH} ]; then
  COURSE_MATERIAL_DB_PATH=""
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
    --use_course_material ${USE_COURSE_MATERIAL} \
    --course_material_path ${COURSE_MATERIAL_PATH} \
    --embedding_model ${EMBEDDING_MODEL} \
    --course_material_db_path ${COURSE_MATERIAL_DB_PATH} \
    --exam-json-path ${file}

  echo "---------------------------------------------------------"
done