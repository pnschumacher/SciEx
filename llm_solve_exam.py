import json
import os
import re
import fitz
from utils import ContentType, CourseMaterialType, prompt_prefix, load_json, stringToBool, write_json_file, write_text_file, info_from_exam_path, process_images, process_context_images, get_index_and_client, delete_index, get_padding_length
import argparse
from types import SimpleNamespace

from llm_clients import OpenAIClient, ClaudeClient, HFTextGenClient, HFLlava
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-type", choices=['openai', 'claude', 'hf_text_gen', 'hf_llava'])
    parser.add_argument("--server-url", default="openai")
    parser.add_argument("--llm-name-full", default="gpt-3.5-turbo-0125")
    parser.add_argument("--llm-name", default='gpt35')
    parser.add_argument("--course-material-path", default=None)
    parser.add_argument("--course-material-type", default=CourseMaterialType.SLIDES)
    parser.add_argument("--embedding-model-name", default="BAAI/bge-large-en")
    parser.add_argument("--embedding-model-path", default=None)
    parser.add_argument("--similarity-top-k", type=int, default=10)
    parser.add_argument("--vector-db-path", default=None)
    parser.add_argument("--transcript-chunk-size", type=int, default=300)
    parser.add_argument("--retrieval-content-type", default=ContentType.TEXT)
    parser.add_argument("--context-content-type", default=ContentType.TEXT)
    parser.add_argument("--exam-json-path")
    parser.add_argument("--use-course-material", type=stringToBool, default=False)
    args = parser.parse_args()

    server_type = args.server_type
    server_url = args.server_url
    llm_name_full = args.llm_name_full
    llm_name = args.llm_name
    course_material_path = args.course_material_path
    course_material_type = args.course_material_type
    embedding_model_name = args.embedding_model_name
    embedding_model_path = args.embedding_model_path
    similarity_top_k = args.similarity_top_k
    vector_db_path = args.vector_db_path
    transcript_chunk_size = args.transcript_chunk_size
    retrieval_content_type = args.retrieval_content_type
    context_content_type = args.context_content_type
    exam_json_path = args.exam_json_path
    use_course_material = args.use_course_material

    if not ("vision" in llm_name_full or "-VL" in llm_name_full) and context_content_type == ContentType.IMAGE:
        raise NotImplementedError("Image context only available for vision models")

    exam_name, lang = info_from_exam_path(exam_json_path)
    if use_course_material:
        if CourseMaterialType.TRANSCRIPTS in course_material_type: 
            out_dir = f"llm_out_{course_material_type}_{similarity_top_k}_{transcript_chunk_size}/{exam_name}"
        else:
            if retrieval_content_type == ContentType.TEXT and context_content_type == ContentType.TEXT:
                out_dir = f"llm_out_{course_material_type}_{similarity_top_k}/{exam_name}"
            else:
                out_dir = f"llm_out_{course_material_type}_{retrieval_content_type}_to_{context_content_type}_{similarity_top_k}/{exam_name}"
        context_path = f"{out_dir}/used_context_{exam_name}_{lang}_{llm_name}.txt"
    else:
        out_dir = f"llm_out/{exam_name}"

    out_path = f"{out_dir}/{exam_name}_{lang}_{llm_name}.txt"

    if os.path.isfile(out_path):
        print("LLM output already available. Skip")
        exit()

    os.makedirs(out_dir, exist_ok=True)

    index = None
    retriever = None
    if use_course_material:
        try:
            index, client = get_index_and_client(
                exam_json_path=exam_json_path, 
                embedding_model_name=embedding_model_name, 
                embedding_model_path=embedding_model_path, 
                course_material_path=course_material_path,
                course_material_type=course_material_type,
                vector_db_path=vector_db_path,
                transcript_chunk_size=transcript_chunk_size,
                retrieval_content_type=retrieval_content_type,
            )
        except FileNotFoundError as e:
            print(e)
            return

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

    prompt = prompt_prefix(lang=lang, use_course_material=use_course_material, context_content_type=context_content_type)
    exam = load_json(f"exams_json/{exam_name}/{exam_name}_{lang}.json")

    exam_out = ''

    if use_course_material:
        used_context = {}

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

            if course_material_type == CourseMaterialType.SLIDES and retrieval_content_type != context_content_type:
                text_nodes_context = []
                context_image_paths = []
                for text_node in text_nodes:
                    filename = text_node.metadata.get("file_name")

                    if filename.endswith(".pdf"):
                        page_number = text_node.metadata.get("page_label")
                        lecture_name = os.path.basename(filename).replace(".pdf", "")
                    elif filename.endswith(".txt"):
                        base_filename = os.path.basename(filename)
                        page_number = int(base_filename.split("-")[-1].replace(".txt", ""))
                        lecture_name = "-".join(base_filename.split("-")[:-1])
                    else:
                        raise NotImplementedError("Only support for .pdf and .txt course material files")
                    
                    format_dir = f"{course_material_path}/{exam_name}/format_files"
                    slide_dir = f"{course_material_path}/{exam_name}/slides"
                    padding_length = get_padding_length(format_dir, lecture_name)
                    page_str_padded = str(page_number).zfill(padding_length)
                                        
                    if context_content_type == ContentType.TEXT:
                        content_directory = slide_dir

                        # All slides have PDF file type except TGI
                        if exam_name == "TGI2324":
                            text_path = os.path.join(content_directory, f"{lecture_name}-{page_str_padded}.txt")
                            with open(text_path, 'r') as file:
                                text = file.read()                    
                        else:
                            pdf_path = os.path.join(content_directory, f"{lecture_name}.pdf")
                            doc = fitz.open(pdf_path)
                            page = doc.load_page(page_number - 1)
                            text = page.get_text()
                            doc.close()

                        latex_pattern = r"<latexit.*?>.*?<\/latexit>"
                        text = re.sub(latex_pattern, "", text)

                        text_node_dict = {
                            "text": text, 
                            "metadata": {
                                "page_label": page_number,
                                "file_name": lecture_name,
                            }
                        }
                        text_nodes_context.append(SimpleNamespace(**text_node_dict))

                    elif context_content_type == ContentType.LAYOUT:
                        content_directory = format_dir

                        text_path = os.path.join(content_directory, f"{lecture_name}-{page_str_padded}.txt")
                        with open(text_path, 'r') as file:
                            text = file.read()

                        text_node_dict = {
                            "text": text, 
                            "metadata": {
                                "page_label": page_number,
                                "file_name": lecture_name,
                            }
                        }
                        text_nodes_context.append(SimpleNamespace(**text_node_dict))

                    elif context_content_type == ContentType.IMAGE:
                        pdf_path = os.path.join(slide_dir, f"{lecture_name}.pdf")
                        output_dir = os.path.join(course_material_path, exam_name, "context_images")
                        os.makedirs(output_dir, exist_ok=True)

                        doc = fitz.open(pdf_path)
                        page = doc.load_page(page_number - 1) 
                        pix = page.get_pixmap(dpi=300)  

                        output_path = os.path.join(output_dir, f"{lecture_name}-{str(page_number).zfill(padding_length)}.png")
                        pix.save(output_path)

                        doc.close()

                        context_image_paths.append(output_path)
                            
            else:
                text_nodes_context = text_nodes

            if text_nodes_context:
                context = [
                    {
                        "Course_Material": text_node.text, 
                        # This could be used to analyze what material is used to answer questions but may introduce noise
                        # "Metadata": {
                        #     "Page": text_node.metadata.get("page_label"), 
                        #     "Filename": text_node.metadata.get("file_name"), 
                        # },
                    } 
                    for text_node in text_nodes_context
                ]
            elif context_image_paths:
                context_image_paths_flatten = [re.sub(r".*?(?=context_images/)", "", path) for path in context_image_paths]
                context = context_image_paths_flatten
            else:
                context = []

            used_context[question_id] = context
            question = {"Context": context, **question}

        print(question)
        out = llm_client.send_request(
            prompt,
            input_body=json.dumps(question),
            images=process_images(exam_name, question) + (process_context_images(context_image_paths) or [])
        )

        print(f'**** Answer: {out}')
        exam_out += f"Answer to Question {question_id}\n"
        exam_out += f"{out}\n"
        exam_out += \
            "\n\n\n\n\n****************************************************************************************\n"
        exam_out += "****************************************************************************************\n\n\n\n\n"

    write_text_file(exam_out, out_path)

    if use_course_material:
        write_json_file(used_context, context_path)
        delete_index(exam_json_path=exam_json_path, client=client)


if __name__ == "__main__":
    main()

