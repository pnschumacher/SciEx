import argparse
import json
import os
from PIL import Image, ImageDraw, ImageFont
import chromadb
import fitz  # PyMuPDF, imported as fitz for backward compatibility reasons
import base64
import io
import re

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


LLM_LIST = ['llama-3.3-slides-only', 'llama-3.3_transcripts-only', 'llama-3.3-slides-and-transcripts']
EXAM_LIST = [
    {"exam_name": "nlp_march_2023", "lang": ["en", "de"]},
    {"exam_name": "dl4cv2_feb_2024", "lang": ["en", "de"]},
    {"exam_name": "dbs_exam_ipd-boehm_2022-ws", "lang": ["de"]},
    {"exam_name": "dbs_exam_ipd-boehm_2023", "lang": ["de"]},
    {"exam_name": "HCI_SS23", "lang": ["en"]},  # de
    {"exam_name": "exam_cg_march_2024", "lang": ["de"]},
    {"exam_name": "ml_4_natural_science", "lang": ["en", "de"]},
    {"exam_name": "TGI2324", "lang": ["de"]},
    {"exam_name": "nlp_march_2024", "lang": ["en"]},  # de
    {"exam_name": "AI2-SoSe-23", "lang": ["en", "de"]},
    {"exam_name": "DLNN-WS2223", "lang": ["en"]},  # de
    {"exam_name": "algo_ws2324", "lang": ["de"]},
]


def map_llm_to_index(llm_name):
    if llm_name not in LLM_LIST:
        raise RuntimeError(f"LLM {llm_name} not in list")
    d = {llm: f"llm{i}" for i, llm in enumerate(LLM_LIST)}
    return d[llm_name]


def map_index_to_llm(index):
    return LLM_LIST[index]


def grading_prompt_prefix(lang, shots=[], with_ref=False, stack_figures=False):
    """
        :param lang: 'en' or 'de'
        :return: prompt prefix as a string
        """
    if stack_figures:
        if lang == 'en':
            extra_message = "Note that the single input figure could contain multiple figures stacked vertically. "
        elif lang == 'de':
            extra_message = "Beachten Sie, dass die einzelne Eingabe Figur mehrere vertikal gestapelte Figuren " \
                            "enthalten kann. "
        else:
            raise RuntimeError(f"No prompt for lang {lang}")
    else:
        extra_message = ""

    if len(shots) == 0:
        shot_message = ""
    else:
        shot_message_en = "Below you are provided with example on how to perform the grading:\n"
        shot_message_de = "Nachfolgend finden Sie ein Beispiel für die Durchführung der Benotung:\n"
        input_w_en = "Input"
        input_w_de = "Eingabe"
        output_w_en = "Output"
        output_w_de = "Ausgabe"

        if lang == 'en':
            shot_message = shot_message_en
            input_w = input_w_en
            output_w = output_w_en
        elif lang == 'de':
            shot_message = shot_message_de
            input_w = input_w_de
            output_w = output_w_de
        else:
            raise RuntimeError(f"No prompt for lang {lang}")

        for shot in shots:
            if with_ref:
                ref_bounds = f"[correct_answer]\n{shot['CorrectAnswer']}\n[/correct_answer] \n"
            else:
                ref_bounds = ""
            shot_message = shot_message + f"{input_w}:\n" \
                                          f"[question]\n{shot['Question']}\n[/question] \n" \
                                          f"[answer]\n{shot['Answer']}\n[/answer] \n" \
                                          f"{ref_bounds}" \
                                          f"[max_score] {shot['MaxScore']} [/max_score] \n" \
                                          f"{output_w}:\n" \
                                          f"[grade] {shot['GoldGrade']} [/grade]\n\n"

    if lang == 'en':
        if with_ref:
            ref_bounds = f"[correct_answer] <correct_answer> [/correct_answer] \n"
            input_ls = "exam question, examinee's answer, correct answer and the maximum possible score"
        else:
            ref_bounds = ""
            input_ls = "exam question, answer and the maximum possible score"
        prompt = f"You are a university professor. Please grade the following exam question. The {input_ls} are provided in the format:\n" \
                 f"[question] <exam_question> [/question] \n" \
                 f"[answer] <answer> [/answer] \n" \
                 f"{ref_bounds}" \
                 f"[max_score] <max_score> [/max_score] \n" \
                 f"The question is provided in JSON format, but the answer can be freeform text. The provided figures in the question (if any) each contains its path at the bottom, which matches the path provided in the JSON. {extra_message}The answer is text-only. If the question asks to draw on the figure, then the answer should contain text description on how the drawing should be." \
                 f"Please provide the grade between [0, <max_score>]. Please provide the reasoning for your grade. Please provide your output in the format: \n" \
                 f"[reason] <reasoning> [/reason] \n" \
                 f"[grade] <grade> [/grade] \n" \
                 f"{shot_message}" \
                 f"Here is your input: \n"
    elif lang == 'de':
        if with_ref:
            ref_bounds = f"[correct_answer] <korrekteAntwort> [/correct_answer] \n"
            input_ls = "Die Prüfungsfrage, die Antwort des Prüflings, die richtige Antwort und die maximal mögliche Punktzahl"
        else:
            ref_bounds = ""
            input_ls = "Die Prüfungsfrage, die Antwort und die maximal mögliche Punktzahl"
        prompt = f"Sie sind Universitätsprofessor. Bitte bewerten Sie die folgende Prüfungsfrage. {input_ls} werden im Format bereitgestellt:\n" \
                 f"[question] <Prüfungsfrage> [/question] \n" \
                 f"[answer] <Antwort> [/answer] \n" \
                 f"{ref_bounds}" \
                 f"[max_score] <maxPunkt> [/max_score] \n" \
                 f"Die Frage wird im JSON-Format bereitgestellt, die Antwort kann jedoch Freiformtext sein. Die bereitgestellten Abbildungen in der Frage (falls vorhanden) enthalten jeweils unten ihren Pfad, der mit dem im JSON bereitgestellten Pfad übereinstimmt. {extra_message}Die Antwort ist nur Text. Wenn es sich bei der Frage darum handelt, auf der Abbildung zu zeichnen, sollte die Antwort eine Textbeschreibung darüber enthalten, wie die Zeichnung aussehen soll." \
                 f"Bitte geben Sie die Note zwischen [0, <maxPunkt>] an. Bitte begründen Sie Ihre Note. Bitte geben Sie Ihre Ausgabe im Format an: \n" \
                 f"[reason] <Grundsatz> [/reason] \n" \
                 f"[grade] <Note> [/grade] \n" \
                 f"{shot_message}" \
                 f"Hier ist Ihre Eingabe: \n"
    else:
        raise RuntimeError(f"No prompt for lang {lang}")

    return prompt


def prompt_prefix(lang, stack_figures=False, use_course_material=False):
    """
    :param lang: 'en' or 'de'
    :return: prompt prefix as a string
    """

    extra_message = ''
    if stack_figures:
        if lang == 'en':
            extra_message += "Note that the single input figure could contain multiple figures stacked vertically. "
        elif lang == 'de':
            extra_message += "Beachten Sie, dass die einzelne Eingabe Figur mehrere vertikal gestapelte Figuren enthalten kann. "
        else:
            raise RuntimeError(f"No prompt for lang {lang}")
    
    if use_course_material:
        if lang == 'en':
            extra_message += "Additionally, you will be provided with some course materials (can be found in 'Context'). " \
                "You can use that additional context to answer the question if it is helpful. " \
                "If it is not helpful, you don't have to use it. " 
        elif lang == 'de':
            extra_message += "Zusätzlich wird Ihnen Kursmaterial zur Verfügung gestellt (zu finden unter 'Context'). " \
                "Sie können diesen zusätzlichen Kontext zur Beantwortung der Frage nutzen, wenn er hilfreich ist und einen Bezug zur Frage hat. " \
                "Wenn nicht, müssen Sie ihn nicht verwenden. " 
        else:
            raise RuntimeError(f"No prompt for lang {lang}")
        
    if lang == 'en':
        prompt = "You are a university student. Please answer the following JSON-formatted exam question. " \
                 "The subquestions (if any) are indexed. " \
                 "The provided figures (if any) each contains its path at the bottom, " \
                 f"which matches the path provided in the JSON. {extra_message}" \
                 "Please give the answers to the question and subquestions that were asked, " \
                 "and index them accordingly in your output. " \
                 "You do not have to provide your output in the JSON format. " \
                 "If you are asked to draw on the figure, then describe with words how you would draw it. " \
                 "Please provide all answers in English. " \
                 "Here is the question: \n"
    elif lang == 'de':
        prompt = "Sie sind Student. Bitte beantworten Sie die folgende JSON-formatierte Prüfungsfrage. " \
                 "Die Unterfragen (falls vorhanden) sind indiziert. " \
                 "Die bereitgestellten Abbildungen (falls vorhanden) enthalten jeweils unten ihren Pfad, " \
                 f"der mit dem im JSON bereitgestellten Pfad übereinstimmt. {extra_message}" \
                 "Bitte geben Sie die Antworten auf die gestellten Fragen " \
                 "und Unterfragen an und indizieren Sie diese in Ihrer Ausgabe entsprechend. " \
                 "Sie müssen Ihre Ausgabe nicht im JSON-Format bereitstellen. " \
                 "Wenn Sie aufgefordert werden, auf der Figur zu zeichnen, beschreiben Sie mit Worten, wie Sie sie zeichnen würden. " \
                 "Bitte geben Sie alle Antworten auf Deutsch an. Hier ist die Frage: \n"
    else:
        raise RuntimeError(f"No prompt for lang {lang}")

    return prompt


def load_json(file_path):
    with open(file_path, 'r') as f:
        obj = json.load(f)
    return obj


def dump_json(obj, file_path):
    with open(file_path, 'w') as f:
        json.dump(obj, f, indent=4)


def info_from_exam_path(exam_json_path):
    exam_name = exam_json_path.split('/')[-2]
    lang = exam_json_path.split('/')[-1].replace('.json', '').split('_')[-1]
    assert os.path.isfile(f"exams_json/{exam_name}/{exam_name}_{lang}.json")
    assert lang in ['en', 'de']
    return exam_name, lang


def collect_figures(question_dict):
    figure_list = []
    if 'Figures' in question_dict:
        figure_list.extend(question_dict['Figures'])
    if 'Subquestions' in question_dict:
        for subquestion_dict in question_dict['Subquestions']:
            if 'Figures' in subquestion_dict:
                figure_list.extend(subquestion_dict['Figures'])
    return figure_list


def add_title(img, title):
    # Create a black bar with the same width as the image and some height for the title
    black_bar_height = 50
    black_bar = Image.new('RGB', (img.width, black_bar_height), color='black')

    # Combine the image with the black bar
    new_img = Image.new('RGB', (img.width, img.height + black_bar_height))
    new_img.paste(img, (0, 0))
    new_img.paste(black_bar, (0, img.height))

    # Draw the title text on the black bar
    draw = ImageDraw.Draw(new_img)
    fontsize = 20
    font = ImageFont.truetype("artifacts/Arial.ttf", size=fontsize)
    text_width = draw.textlength(title, font=font)
    text_height = fontsize
    text_position = ((img.width - text_width) // 2, img.height + (black_bar_height - text_height) // 2)
    draw.text(text_position, title, fill='white', font=font)

    return new_img


def pdf2pil(pdf_path):
    images = []
    doc = fitz.open(pdf_path)  # open document
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=300)  # render page to an image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images


def combine_images(images):
    # Find the maximum width and height among all images
    max_width = max(int(image.width*1.1) for image in images)
    height_gap = max(int(image.height*0.05) for image in images)
    total_height = sum(image.height for image in images) + height_gap * (len(images) + 1)

    # Create a new blank image with the maximum width and height
    combined_image = Image.new('RGB', (max_width, total_height))

    # Paste each image onto the blank image, padding if necessary
    y_offset = height_gap
    for image in images:
        x_offset = (max_width - image.width) // 2  # calculate horizontal padding
        combined_image.paste(image, (x_offset, y_offset))
        y_offset = y_offset + image.height + height_gap

    return combined_image


def load_images(image_paths):
    """
    :param image_paths: path to the images. can be pdf file containing multiple pages
    :return: images: list of single images loaded by PIL. images_paths_flatten: path of each single image
    """
    images = []
    images_paths_flatten = []
    for path in image_paths:
        if path.endswith('.pdf'):
            images_tmp = pdf2pil(path)
            images.extend(images_tmp)
            images_paths_flatten.extend([path] * len(images_tmp))
        else:
            images.append(Image.open(path))
            images_paths_flatten.append(path)
    return images, images_paths_flatten


def write_text_file(string, file_path):
    with open(file_path, 'w') as file:
        file.write(string)


def encode_image(image_path=None, pil_image=None):
    if image_path is not None:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    if pil_image is not None:
        # Create an in-memory binary stream (bytes-like object)
        image_stream = io.BytesIO()
        # Save the PIL image to the in-memory stream in PNG format
        pil_image.save(image_stream, format='PNG')
        # Seek to the beginning of the stream
        image_stream.seek(0)
        # Encode the image data as base64 and decode it to convert from bytes to string
        encoded_image = base64.b64encode(image_stream.read()).decode('utf-8')
        return encoded_image


def process_images(exam_name, question):
    image_paths = collect_figures(question)
    image_full_paths = [f"exams_json/{exam_name}/{x}" for x in image_paths]
    images, image_full_paths_flatten = load_images(image_full_paths)
    image_paths_flatten = [path.replace(f"exams_json/{exam_name}/", "") for path in image_full_paths_flatten]
    image_titles = [f"Figure: {x}" for x in image_paths_flatten]
    images = [add_title(img, title) for img, title in zip(images, image_titles)]
    return images


def extract_answer(question_id, answer):
    start_marker = f"Answer to Question {question_id}"
    end_marker = "****************************************************************************************\n" \
                 "****************************************************************************************"

    start_index = answer.find(start_marker)
    if start_index == -1:
        raise RuntimeError(f"Problem with extracting answer for question {question_id}")

    end_index = answer.find(end_marker, start_index)
    if end_index == -1:
        raise RuntimeError(f"Problem with extracting answer for question {question_id}")

    return answer[start_index + len(start_marker):end_index]


def parse_grade(llm_out, max_score):
    grade_regex = r"\[grade\]\s*(\d+(?:[.,]\d+)?)"
    matches = re.findall(grade_regex, llm_out)
    grade = parse_matched_float(matches, max_score)
    if grade is not None:
        return grade

    grade_regex = r"(\d+(?:[.,]\d+)?)"
    matches = re.findall(grade_regex, llm_out)
    grade = parse_matched_float(matches, max_score)
    return grade


def parse_matched_float(matches, max_score):
    for match in reversed(matches):
        grade_str = match
        # Replace comma with dot if present
        grade_str = grade_str.replace(',', '.')
        try:
            grade = float(grade_str.strip())
            if grade >= 0 and grade <= max_score:  # Assuming max_score is defined elsewhere
                return grade
        except RuntimeError:
            continue
    return None


def load_text_file(file_path, single_str=True):
    """
    Load the whole text file to a single string or to a list of strings, each represents a line
    """
    if single_str:
        with open(file_path, "r") as file:
            string = file.read()
        return string
    else:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        return lines


def remove_key(d, key):
    new_d = d.copy()
    new_d.pop(key)
    return new_d


def get_index(exam_json_path, embedding_model_name, course_material_path):
    exam_name, lang = info_from_exam_path(exam_json_path)

    print("Creating new index...")
    
    embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)

    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection(f"{exam_name}_{lang}")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    if not os.path.exists(f"{course_material_path}/{exam_name}_{lang}"):
        raise FileNotFoundError(f"Course material path {course_material_path}/{exam_name}_{lang} does not exist.")

    slide_directory = f"{course_material_path}/{exam_name}_{lang}/slides"
    transcript_directory = f"{course_material_path}/{exam_name}_{lang}/transcripts"

    slide_nodes = []
    transcript_nodes = []

    if os.path.exists(slide_directory):
        slide_documents = SimpleDirectoryReader(slide_directory).load_data()

        # TODO: Do not hardcode this on NLP exams
        slide_prefix_pattern = r'^[^\n]*Niehues[^\n]*\n'
        date_slide_pattern_en = r"\n(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\d{1,3}"
        date_slide_pattern_de = r"\n\d{1,2}. (Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember) \d{4}\d{1,3}"
        latex_pattern = r"<latexit.*?>.*?<\/latexit>"

        for doc in slide_documents:
            new_text = re.sub(slide_prefix_pattern, "", doc.text)
            new_text = re.sub(date_slide_pattern_en, "", new_text)
            new_text = re.sub(date_slide_pattern_de, "", new_text)
            new_text = re.sub(latex_pattern, "", new_text)

            doc.text_resource.text = new_text

        slide_window_documents = []
        window_size = 5

        for i in range(len(slide_documents) - window_size):
            document_text = ""
            for j in range(window_size):
                document_text += slide_documents[i + j].text
            
            slide_window_document = slide_documents[i].model_copy()
            slide_window_document.text_resource.text = document_text
            slide_window_documents.append(slide_window_document)

        # This ensures that each node is a separate slide
        slide_splitter = SentenceSplitter(chunk_size=100000, chunk_overlap=0)
        slide_nodes = slide_splitter.get_nodes_from_documents(documents=slide_window_documents)
            
    if os.path.exists(transcript_directory):
        transcript_documents = SimpleDirectoryReader(transcript_directory).load_data()

        text_splitter = SentenceSplitter(chunk_size=200, chunk_overlap=10)
        transcript_nodes = text_splitter.get_nodes_from_documents(documents=transcript_documents)

    nodes = slide_nodes + transcript_nodes

    if not nodes:
        print("No nodes were created")
        return

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)

    return index


def stringToBool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "yes", "1"):
        return True
    elif value.lower() in ("false", "no", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")