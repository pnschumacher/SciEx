#!/bin/bash
set -eu  # Crash if variable used without being set

source venv/bin/activate
source env_vars.sh
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH # this is done so python uses correct sqlite3 version

SERVER_TYPE=$1
LLM_NAME=$2
LLM_NAME_FULL=$3
SERVER_URL=$4
NR_SHOT=$5
SHOT_TYPE=$6
REF=$7


if [ -z ${SERVER_TYPE} ]; then
  SERVER_TYPE="openai"
fi
if [ -z ${LLM_NAME} ]; then
  LLM_NAME="gpt4v"
fi
if [ -z ${LLM_NAME_FULL} ]; then
  LLM_NAME_FULL=gpt-4-vision-preview
fi
if [ -z ${SERVER_URL} ]; then
  SERVER_URL="openai"
fi
if [ -z ${NR_SHOT} ]; then
  NR_SHOT=1
fi
if [ -z ${SHOT_TYPE} ]; then
  SHOT_TYPE="same_question"
fi

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
