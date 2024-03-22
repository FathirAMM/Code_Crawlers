import os
import uuid
import base64
from IPython import display
from unstructured.partition.pdf import partition_pdf
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.schema.document import Document
from langchain.vectorstores import FAISS
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import numpy as np
from openai import OpenAI
import unstructured
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI


openai_api_key = "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"

prompt_template = """
You are document analyst tasking with providing insights from documents.
Answer the question based only on the following context, which can include text, images and tables:
{context}
Question: {question}
Just return the helpful answer in as much as detailed possible.
Answer:
"""
output_path = "./images3"


def process_all_pdfs_to_vector_db(directory_path):
    output_path = "./images3"
    raw_pdf_elements = []

    # Iterate over each file in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):  # Check if the file is a PDF
            # Construct the full path to the PDF file
            pdf_file_path = os.path.join(directory_path, filename)

            # Process the PDF and append its elements to raw_pdf_elements
            elements = partition_pdf(
                filename=pdf_file_path,
                extract_images_in_pdf=True,
                infer_table_structure=True,
                chunking_strategy="by_title",
                max_characters=4000,
                new_after_n_chars=3800,
                combine_text_under_n_chars=2000,
                extract_image_block_output_dir=output_path,
            )
            raw_pdf_elements.append(elements)
            print("raw elements created")

    return raw_pdf_elements


# Categorize elements by type
def categorize_elements(raw_pdf_elements):
    """
    Categorize extracted elements from a PDF into tables and texts.
    raw_pdf_elements: List of unstructured.documents.elements
    """
    tables = []
    texts = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))

    # Optional: Enforce a specific token size for texts
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=4000, chunk_overlap=0
    )
    joined_texts = " ".join(texts)
    texts_4k_token = text_splitter.split_text(joined_texts)
    print("elements categroized")
    return texts_4k_token, tables, texts


def generate_text_summaries(texts, tables, summarize_texts=False):
    """
    Summarize text elements
    texts: List of str
    tables: List of str
    summarize_texts: Bool to summarize texts
    """

    # Prompt
    prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval. \
        These summaries will be embedded and used to retrieve the raw text or table elements. \
        Give a concise summary of the table or text that is well optimized for retrieval. Table or text: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Text summary chain
    model = ChatOpenAI(temperature=0, model="gpt-4", api_key=openai_api_key)
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    # Initialize empty summaries
    text_summaries = []
    table_summaries = []

    # Apply to text if texts are provided and summarization is requested
    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
    elif texts:
        text_summaries = texts

    # Apply to tables if tables are provided
    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
    print("text,table summary finished")

    return text_summaries, table_summaries


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def summarize_image(encoded_image):
    prompt = [
        SystemMessage(
            content="You are document analyst tasking with providing insights from documents"
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "You are document analyst tasking with providing insights from documents.\n"
                    "You will be given a mixed of text, tables, and image(s) usually of charts or graphs.\n"
                    "Use this information to provide insights related to the user question. \n",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                },
            ]
        ),
    ]
    response = ChatOpenAI(
        model="gpt-4-vision-preview", openai_api_key=openai_api_key, max_tokens=1024
    ).invoke(prompt)
    return response.content


def process_pdf_to_vector_db(directory_path):

    raw_elements = process_all_pdfs_to_vector_db(directory_path)
    texts_4k_token, tables, texts = categorize_elements(raw_elements)
    text_summaries, table_summaries = generate_text_summaries(
        texts_4k_token, tables, summarize_texts=True
    )
    # Get image summaries
    image_elements = []
    image_summaries = []

    for i in os.listdir(output_path):
        if i.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(output_path, i)
            encoded_image = encode_image(image_path)
            image_elements.append(encoded_image)
            summary = summarize_image(encoded_image)
            image_summaries.append(summary)

    # Create Documents and Vectorstore
    documents = []
    retrieve_contents = []

    for e, s in zip(texts, text_summaries):
        i = str(uuid.uuid4())
        doc = Document(
            page_content=s, metadata={"id": i, "type": "text", "original_content": e}
        )
        retrieve_contents.append((i, e))
        documents.append(doc)

    for e, s in zip(tables, table_summaries):
        doc = Document(
            page_content=s, metadata={"id": i, "type": "table", "original_content": e}
        )
        retrieve_contents.append((i, e))
        documents.append(doc)

    for e, s in zip(image_elements, image_summaries):
        doc = Document(
            page_content=s, metadata={"id": i, "type": "image", "original_content": e}
        )
        retrieve_contents.append((i, s))
        documents.append(doc)

    vectorstore = FAISS.from_documents(
        documents=documents, embedding=OpenAIEmbeddings(openai_api_key=openai_api_key)
    )
    vectorstore.save_local("faiss_index_chat")

    return vectorstore


# final function - we neeed to call this function in main file
def answer_pdf(question, directory_path):
    vectorstore = process_pdf_to_vector_db(directory_path)
    relevant_docs = vectorstore.similarity_search(question)
    context = ""
    relevant_images = []

    qa_chain = LLMChain(
        llm=ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key, max_tokens=1024),
        prompt=PromptTemplate.from_template(prompt_template),
    )

    for d in relevant_docs:
        if d.metadata["type"] == "text":
            context += "[text]" + d.metadata["original_content"]
        elif d.metadata["type"] == "table":
            context += "[table]" + d.metadata["original_content"]
        elif d.metadata["type"] == "image":
            context += "[image]" + d.page_content
            relevant_images.append(d.metadata["original_content"])
    result = qa_chain.run({"context": context, "question": question})

    return result, relevant_images


def answer_only(question):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.load_local("faiss_index_chat", embeddings)
    relevant_docs = vectorstore.similarity_search(question)
    context = ""
    relevant_images = []

    qa_chain = LLMChain(
        llm=ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key, max_tokens=2048),
        prompt=PromptTemplate.from_template(prompt_template),
    )

    for d in relevant_docs:
        if d.metadata["type"] == "text":
            context += "[text]" + d.metadata["original_content"]
        elif d.metadata["type"] == "table":
            context += "[table]" + d.metadata["original_content"]
        elif d.metadata["type"] == "image":
            context += "[image]" + d.page_content
            relevant_images.append(d.metadata["original_content"])
    result = qa_chain.run({"context": context, "question": question})

    return result, relevant_images
