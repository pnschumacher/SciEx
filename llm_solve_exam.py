import json
import os
from utils import prompt_prefix, load_json, stringToBool, write_text_file, info_from_exam_path, process_images, get_index
import argparse

from llm_clients import OpenAIClient, ClaudeClient, HFTextGenClient, HFLlava
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-type", choices=['openai', 'claude', 'hf_text_gen', 'hf_llava'])
    parser.add_argument("--server-url", default="openai")
    parser.add_argument("--llm-name-full", default="gpt-3.5-turbo-0125")
    parser.add_argument("--llm-name", default='gpt35')
    parser.add_argument("--course-material-path", default=None)
    parser.add_argument("--embedding-model-name", default="BAAI/bge-large-en")
    parser.add_argument("--similarity-top-k", type=int, default=10)
    parser.add_argument("--exam-json-path")
    parser.add_argument("--use-course-material", type=stringToBool, default=False)
    args = parser.parse_args()

    server_type = args.server_type
    server_url = args.server_url
    llm_name_full = args.llm_name_full
    llm_name = args.llm_name
    course_material_path = args.course_material_path
    embedding_model_name = args.embedding_model_name
    similarity_top_k = args.similarity_top_k
    exam_json_path = args.exam_json_path
    use_course_material = args.use_course_material

    index = None
    retriever = None
    if use_course_material:
        index = get_index(exam_json_path, embedding_model_name, course_material_path)
        retriever = VectorIndexRetriever(index=index, similarity_top_k=similarity_top_k)
    
    if server_type == 'openai':
        llm_client = OpenAIClient(model=llm_name_full, server_url=server_url, seed=0)
    elif server_type == 'claude':
        llm_client = ClaudeClient(model=llm_name_full)
    elif server_type == 'hf_text_gen':
        llm_client = HFTextGenClient(model=llm_name_full, server_url=server_url)
    elif server_type == 'hf_llava':
        llm_client = HFLlava(model=llm_name_full, device='cuda')
    else:
        raise RuntimeError(f"server_type {server_type} not implemented.")

    exam_name, lang = info_from_exam_path(exam_json_path)

    if use_course_material:
        out_dir = f"llm_out_cm/{exam_name}"
    else:
        out_dir = f"llm_out/{exam_name}"

    out_path = f"{out_dir}/{exam_name}_{lang}_{llm_name}.txt"

    if os.path.isfile(out_path):
        print("LLM output already available. Skip")
        exit()

    os.makedirs(out_dir, exist_ok=True)

    prompt = prompt_prefix(lang=lang, use_course_material=use_course_material)
    exam = load_json(f"exams_json/{exam_name}/{exam_name}_{lang}.json")

    exam_out = ''
    for question in exam['Questions']:
        question_id = question.pop("Index")

        if retriever:
            question_content = question.get("Description", "")
            sub_questions = question.get("Subquestions", [])
            for sub_question in sub_questions:
                sub_content = sub_question.get("Content", "")
                question_content += "\n" + sub_content
                
            related = retriever.retrieve(question_content)
            text_nodes = [node_with_score.node for node_with_score in related]

            context = [
                {
                    "Course_Material": text_node.text, 
                    # This could be used to analyze what material is used to answer questions but may introduce noise
                    # "Metadata": {
                    #     "Page": text_node.metadata.get("page_label"), 
                    #     "Filename": text_node.metadata.get("file_name"), 
                    # },
                } 
                for text_node in text_nodes
            ]

            question = {"Context": context, **question}

        print(question)
        out = llm_client.send_request(
            prompt,
            input_body=json.dumps(question),
            images=process_images(exam_name, question)
        )

        print(f'**** Answer: {out}')
        exam_out += f"Answer to Question {question_id}\n"
        exam_out += f"{out}\n"
        exam_out += \
            "\n\n\n\n\n****************************************************************************************\n"
        exam_out += "****************************************************************************************\n\n\n\n\n"

    write_text_file(exam_out, out_path)


if __name__ == "__main__":
    main()

