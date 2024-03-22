import os
import numpy as np
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
from openai import OpenAI
from langchain.vectorstores import Pinecone as pine
from pinecone import Pinecone
from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import SeleniumURLLoader


openai_api_key = ""

# initializing pinecone index
pc = Pinecone(api_key="")
index = pc.Index("multimodal")

openai_client = OpenAI(api_key=openai_api_key)
MODEL = "text-embedding-ada-002"


def clean_text_from_csv_loader(data):
    clean_data = []

    for document in data:
        page_content = document.page_content
        soup = BeautifulSoup(page_content, "html.parser")
        text_content = soup.get_text(separator="\n", strip=True)
        # print(text_content)
        clean_document = {
            "page_content": text_content,
            # feedback': document.metadata.get('Feedback', '')
        }
        # for document in cleaned_data:

        clean_data.append(clean_document)
    clean_data = clean_data[0]["page_content"]
    return clean_data


# Generate summaries of text elements
def generate_text_summaries(texts):
    """
    Summarize text elements
    summarize_texts: Bool to summarize texts
    """

    # Prompt
    prompt_text = """You are an assistant tasked with summarizing  text for retrieval. \
    These summaries will be embedded and used to retrieve the raw texts. \
    Give a concise summary of the  text that is well optimized for retrieval. Table or text: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Text summary chain
    model = ChatOpenAI(temperature=0, model="gpt-4", api_key=openai_api_key)
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    # Initialize empty summaries
    text_summaries = []

    # Apply to text if texts are provided and summarization is requested
    if texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
    elif texts:
        text_summaries = texts

    return text_summaries


def create_faiss_db(cleaned_data, text_summaries):
    documents = []
    retrieve_contents = []

    for e, s in zip(cleaned_data, text_summaries):
        i = str(uuid.uuid4())
        doc = Document(
            page_content=s, metadata={"id": i, "type": "text", "original_content": e}
        )
        retrieve_contents.append((i, e))
        documents.append(doc)
    vectorstore = FAISS.from_documents(
        documents=documents, embedding=OpenAIEmbeddings(openai_api_key=openai_api_key)
    )
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.load_local("faiss_index", embeddings)
    faiss_db = db.merge_from(vectorstore)
    print("injected to faiss")

    return faiss_db


def create_summaries(docs):
    cleaned_data = clean_text_from_csv_loader(docs)
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=2000, chunk_overlap=0
    )

    texts_4k_token = text_splitter.split_text(cleaned_data)
    text_summaries = generate_text_summaries(texts_4k_token)
    return text_summaries, cleaned_data


# final function for pdf
def inject_to_faiss_csv(file_path):
    loader = UnstructuredCSVLoader(file_path, mode="elements")
    docs = loader.load()
    text_summaries, cleaned_data = docs
    faiss_db = create_faiss_db(cleaned_data, text_summaries)
    return faiss_db


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


# final function to pinecone


# inject to pinecone db
def inject_to_pinecone_csv(file_path):
    loader = UnstructuredCSVLoader(file_path, mode="elements")
    docs = loader.load()
    text_summaries, cleaned_data = create_summaries(docs)
    generate_embeddings_and_upsert(text_summaries)
    print("injected to pinecone db")


def inject_csv(file_path, vector_db):
    if vector_db == "faiss":
        vector_db = inject_to_faiss_csv(file_path)
    else:
        vector_db = inject_to_pinecone_csv(file_path)
    return vector_db


def inject_url(url, vector_db):
    urls = [url]
    loader = SeleniumURLLoader(urls=urls)
    data = loader.load()
    text_summaries, cleaned_data = create_summaries(data)
    if vector_db == "faiss":
        vector_database = create_faiss_db(cleaned_data, text_summaries)
    else:
        vector_database = generate_embeddings_and_upsert(text_summaries)
    return vector_database
