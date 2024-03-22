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
from pinecone import Pinecone
from langchain.vectorstores import Pinecone as pine
import numpy as np
from langchain_community.chat_models import ChatAnthropic


openai_api_key = ""

prompt_template = """
You are financial analyst tasking with providing investment advice.
Answer the question based only on the following context, which can include text, images and tables:
{context}
Question: {question}
Just return the helpful answer in as much as detailed possible.
Answer:
"""


# loading faiss db
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = FAISS.load_local("faiss_index", embeddings)

# loading pinecone db
pc = Pinecone(api_key="")
index = pc.Index("multimodal")
model_name = "text-embedding-ada-002"
embed = OpenAIEmbeddings(model=model_name, openai_api_key=openai_api_key)
text_field = "text"
pinecone_vectorstore = pine(index, embed.embed_query, text_field)


def answer_question_own_model(question, model, database_name, api_key):
    # Default assignment to prevent UnboundLocalError
    vector_db = None

    if database_name == "faiss":
        vector_db = (
            db  # Ensure 'db' is defined earlier in your code or passed as an argument
        )
    elif database_name == "pinecone":
        vector_db = (
            pinecone_vectorstore  # Ensure 'pinecone_vectorstore' is defined or passed
        )

    # Handle case where vector_db is not assigned
    if vector_db is None:
        raise ValueError(f"Unsupported database_name: {database_name}")

    qa_chain = LLMChain(
        llm=ChatOpenAI(model=model, openai_api_key=api_key, max_tokens=1024),
        prompt=PromptTemplate.from_template(prompt_template),
    )
    relevant_docs = vector_db.similarity_search(question)
    context = ""
    relevant_images = []

    for d in relevant_docs:
        if database_name == "faiss":
            if d.metadata["type"] == "text":
                context += "[text]" + d.metadata["original_content"]
            elif d.metadata["type"] == "table":
                context += "[table]" + d.metadata["original_content"]
            elif d.metadata["type"] == "image":
                context += "[image]" + d.page_content
                relevant_images.append(d.metadata["original_content"])
        elif database_name == "pinecone":
            context += "[text]" + d.page_content
            relevant_images = None

    result = qa_chain.run({"context": context, "question": question})
    return result, relevant_images
