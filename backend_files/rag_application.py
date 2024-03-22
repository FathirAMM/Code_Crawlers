# solved pine cone issue

import os
import uuid
import base64
from IPython import display
from unstructured.partition.pdf import partition_pdf
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings as fe
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
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings as pe
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_together import TogetherEmbeddings
from langchain_community.llms import Together
from googlesearch import search

openai_api_key = "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"
# claude_api_key = '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'
claude_api_key = "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"
together_api_key = "d4c44d27bf2e35b7550df54a1cd6ef5473c0d19a42d3c1435799951e323098a5"

prompt_template = """
You are financial analyst tasking with providing investment advice.
Answer the question based only on the following context, which can include text, images and tables:
{context}
Question: {question}
Just return the helpful answer in as much as detailed possible.
Answer:
"""

# loading faiss db
embeddings = fe(openai_api_key=openai_api_key)
# db = FAISS.load_local("faiss_index", embeddings)
db = FAISS.load_local("faiss_index_new", embeddings)


# loading pinecone db
pc = Pinecone(api_key="f3188074-c2dc-4598-8565-768d56c2dfff")
index = pc.Index("multimodal")
model_name = "text-embedding-ada-002"
embed = pe(model=model_name, openai_api_key=openai_api_key)
text_field = "text"
pinecone_vectorstore = PineconeVectorStore(index, embeddings, text_field)

# query = "hello"
# pinecone_vectorstore.similarity_search(query,k=3)


# initialize the model
def initialize_model(model_name):
    if model_name == "gpt-4":
        return LLMChain(
            llm=ChatOpenAI(
                model="gpt-4", openai_api_key=openai_api_key, max_tokens=1024
            ),
            prompt=PromptTemplate.from_template(prompt_template),
        )
    elif model_name == "gpt-3.5-turbo":
        return LLMChain(
            llm=ChatOpenAI(
                model="gpt-3.5-turbo", openai_api_key=openai_api_key, max_tokens=1024
            ),
            prompt=PromptTemplate.from_template(prompt_template),
        )
    elif model_name == "claude-2":
        return LLMChain(
            llm=ChatAnthropic(
                temperature=0, anthropic_api_key=claude_api_key, model_name="claude-2"
            ),
            prompt=PromptTemplate.from_template(prompt_template),
        )
    elif model_name == "mistral":
        return LLMChain(
            llm=Together(
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                temperature=0,
                max_tokens=500,
                top_k=3,
                together_api_key=together_api_key,
            ),
            prompt=PromptTemplate.from_template(prompt_template),
        )
    elif model_name == "llama2-7b":
        return LLMChain(
            llm=Together(
                model="meta-llama/Llama-2-7b-chat-hf",
                temperature=0,
                max_tokens=500,
                top_k=3,
                together_api_key=together_api_key,
            ),
            prompt=PromptTemplate.from_template(prompt_template),
        )
    elif model_name == "gemma-7b":
        return LLMChain(
            llm=Together(
                model="google/gemma-7b-it",
                temperature=0,
                max_tokens=500,
                top_k=3,
                together_api_key=together_api_key,
            ),
            prompt=PromptTemplate.from_template(prompt_template),
        )

    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def answer_question(question, model, database_name):
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

    qa_chain = initialize_model(
        model
    )  # Ensure this function is defined and works as expected
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


def get_urls(query):
    results = search(query)
    i = 0
    urls = []
    for url in results:
        while i < 5:
            i = i + 1
            urls.append(url)
            # print(url)
        break
    return urls
