OUTPUT_DIR=${5:-"llm_out_cm"} 
FILTERED_OUTPUT_DIR=${7:-"llm_out_cm_filtered"}

if [ ! -d "$FILTERED_OUTPUT_DIR" ]; then
  mkdir "$FILTERED_OUTPUT_DIR"
fi

declare -A mapping

mapping["llama3.3_transcript"]="llm12"
mapping["qwen2.5_transcript"]="llm13"

find "$OUTPUT_DIR" \( -name "used_context*" -prune \) -o -mindepth 1 -print | while IFS= read -r file; do
    relpath="${file#$OUTPUT_DIR/}"

    new_relpath="$relpath"
    for key in "${!mapping[@]}"; do
        new_relpath="${new_relpath//$key/${mapping[$key]}}"
    done

    dest="$FILTERED_OUTPUT_DIR/$new_relpath"

    if [ -d "$file" ]; then
         mkdir -p "$dest"
    else
         mkdir -p "$(dirname "$dest")"
         cp "$file" "$dest"
    fi
done

