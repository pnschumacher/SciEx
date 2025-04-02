OUTPUT_DIR=${5:-"llm_out_slides_1"} 
FILTERED_OUTPUT_DIR=${7:-"llm_out_slides_1_filtered"}

if [ ! -d "$FILTERED_OUTPUT_DIR" ]; then
  mkdir "$FILTERED_OUTPUT_DIR"
fi

declare -A mapping

mapping["llama3.3"]="llm8"
mapping["qwen2.5"]="llm9"
mapping["llama3.1"]="llm10"
mapping["qwen2vl"]="llm11"

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

