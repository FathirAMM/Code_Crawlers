import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.chat_models import ChatOpenAI
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_recall,
    context_precision,
)
from pinecone import Pinecone
from langchain.vectorstores import Pinecone as pine
from langchain.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


def evaluate_rag_using_ragas(
    openai_api_key: str,
    model: str,
    file_path: str = "/content/drive/MyDrive/LLM evaluation with RAGAS/test_df_for_ft.csv",
):

    os.environ["OPENAI_API_KEY"] = openai_api_key

    test_df = pd.read_csv(file_path)

    test_questions = test_df["question"].values.tolist()
    test_groundtruths = test_df["ground_truth"].values.tolist()

    prompt_template = """
    You are document analyst tasking with providing insights from documents.
    Answer the question based only on the following context, which can include text, images and tables:
    {context}
    Question: {question}
    Just return the helpful answer in as much as detailed possible.
    Answer:
    """

    qa_chain = LLMChain(
        llm=ChatOpenAI(
            model=model, openai_api_key=openai_api_key, max_tokens=1024, temperature=0
        ),
        prompt=PromptTemplate.from_template(prompt_template),
    )

    model_name = "text-embedding-ada-002"
    embeddings = OpenAIEmbeddings(model=model_name, openai_api_key=openai_api_key)

    # loading pinecone db
    pc = Pinecone(api_key="f3188074-c2dc-4598-8565-768d56c2dfff")
    index = pc.Index("multimodal")
    model_name = "text-embedding-ada-002"
    embed = OpenAIEmbeddings(model=model_name, openai_api_key=openai_api_key)
    text_field = "text"
    vector_db = PineconeVectorStore(index, embeddings, text_field)

    answers = []
    contexts = []
    for question in test_questions:
        relevant_docs = vector_db.similarity_search(question, k=3)
        context = []
        for d in relevant_docs:
            context.append(d.page_content)

        response = qa_chain.run({"context": context, "question": question})
        answers.append(response)
        contexts.append(context)

    response_dataset = Dataset.from_dict(
        {
            "question": test_questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": test_groundtruths,
        }
    )

    metrics = [
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
        answer_correctness,
    ]

    results = evaluate(response_dataset, metrics, raise_exceptions=False)
    return results
