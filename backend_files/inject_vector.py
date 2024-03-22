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
from langchain.schema.document import Document
from langchain.vectorstores import FAISS
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pinecone import Pinecone
import numpy as np
from openai import OpenAI
from langchain.vectorstores import Pinecone as pine
import tempfile
import shutil

# openai_key
openai_api_key = "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"

# initializing pinecone index
pc = Pinecone(api_key="f3188074-c2dc-4598-8565-768d56c2dfff")
index = pc.Index("multimodal")

openai_client = OpenAI(api_key=openai_api_key)
MODEL = "text-embedding-ada-002"


# categorize the elements by tables and texts
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
    print("elements categorized as texts and tables")
    return texts, tables


# generate text table summaries
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

    print("text and table summaries generated")
    return text_summaries, table_summaries


# encode image
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# generate image summaries
def summarize_image(encoded_image):
    prompt = [
        SystemMessage(
            content="You are financial analyst tasking with providing investment advice"
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "You are financial analyst tasking with providing investment advice.\n"
                    "You will be given a mixed of text, tables, and image(s) usually of charts or graphs.\n"
                    "Use this information to provide investment advice related to the user question. \n",
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


# creating text,table, image summaries
def creating_summaries(file_path):

    output_path = "./images_new2"
    print(f"Temporary directory created: {output_path}")
    raw_pdf_elements = partition_pdf(
        filename=file_path,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        extract_image_block_output_dir=output_path,
    )
    texts, tables = categorize_elements(raw_pdf_elements)

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=4000, chunk_overlap=0
    )
    joined_texts = " ".join(texts)
    texts_4k_token = text_splitter.split_text(joined_texts)

    text_summaries, table_summaries = generate_text_summaries(
        texts_4k_token, tables, summarize_texts=True
    )
    image_elements = []
    image_summaries = []
    for i in os.listdir(output_path):
        if i.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(output_path, i)
            encoded_image = encode_image(image_path)
            image_elements.append(encoded_image)
            summary = summarize_image(encoded_image)
            image_summaries.append(summary)
    print("all summaries created")
    # Remove all elements in the output_path
    for filename in os.listdir(output_path):
        file_path = os.path.join(output_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
    return (
        texts,
        text_summaries,
        tables,
        table_summaries,
        image_elements,
        image_summaries,
    )


# add summaries to the vector db
def add_new_vectors_to_faiss(
    texts, text_summaries, tables, table_summaries, image_elements, image_summaries
):
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
    vectorstore.save_local("faiss_index_new")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db_new = FAISS.load_local("faiss_index_new", embeddings)
    db = FAISS.load_local("faiss_index", embeddings)
    db = db.merge_from(db_new)

    return db_new


# inject new data into faiss db using one function
def inject_to_faiss(file_path):

    texts, text_summaries, tables, table_summaries, image_elements, image_summaries = (
        creating_summaries(file_path)
    )
    db = add_new_vectors_to_faiss(
        texts, text_summaries, tables, table_summaries, image_elements, image_summaries
    )
    print("injected to faiss")
    return db


# generate embedings for pinecone
def generate_embeddings_and_upsert(texts):
    batch_size = 10  # Adjust based on your rate limits and testing
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        # Generate embeddings for the batch
        res = openai_client.embeddings.create(input=batch_texts, model=MODEL)
        embeddings = [np.array(record.embedding) for record in res.data]

        # Upsert embeddings to Pinecone with metadata
        vectors = []
        for text, embedding in zip(batch_texts, embeddings):
            vectors.append(
                {
                    "id": str(uuid.uuid4()),  # Generate a unique ID for each vector
                    "values": embedding.tolist(),  # Convert embedding to list
                    "metadata": {"text": text},  # Include original text as metadata
                }
            )
        index.upsert(vectors=vectors)


# inject to pinecone db
def inject_to_pinecone(file_path):
    texts, text_summaries, tables, table_summaries, image_elements, image_summaries = (
        creating_summaries(file_path)
    )
    all_summaries = table_summaries + text_summaries + image_summaries
    generate_embeddings_and_upsert(all_summaries)
    print("injected to pinecone db")


def inject_image(image_path, vector_db):
    pass
