import openai
import os
import json
import tiktoken
import numpy as np
import pandas as pd
import random
from collections import defaultdict
from openai import OpenAI




# Define input and output file paths
input_file = "finetune.jsonl"
train_output_file = "train_set.jsonl"
validation_output_file = "validation_set.jsonl"


def create_dataset(question, answer):
    DEFAULT_SYSTEM_PROMPT = "You are financial analyst tasking with providing investment advice and insights and designed to handle inquiries related finance. Your task is to assist the user in addressing their queries effectively."
    return {
        "messages": [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            },  # System prompt message
            {"role": "user", "content": question},  # User's question
            {"role": "assistant", "content": answer},  # Assistant's answer
        ]
    }


def generate_dataset(file_path):
    df = pd.read_csv(file_path)
    # Open a JSONL file for writing
    with open("finetune.jsonl", "w") as f:
        # Iterate over each row in the DataFrame
        for _, row in df.iterrows():
            # Create a dataset from the question and answer in the current row
            example_str = json.dumps(create_dataset(row["Question"], row["Answer"]))
            # Write the dataset to the JSONL file
            f.write(example_str + "\n")
            print("dataset generated")


# Function to split the JSONL file into train and validation sets
def split_train_validation(input_file, train_ratio=0.8):
    # Read the JSONL file
    with open(input_file, "r") as f:
        lines = f.readlines()

    # Shuffle the lines randomly
    random.shuffle(lines)

    # Calculate the number of lines for the train set
    train_size = int(len(lines) * train_ratio)

    # Split the lines into train and validation sets
    train_set = lines[:train_size]
    validation_set = lines[train_size:]
    print("dataset splitted")
    return train_set, validation_set


def fine_tuning(
    train_output_file,
    validation_output_file,
    train_set,
    validation_set,
    suffix_name,
    api_key,
):
    # Write the train set to a new file
    with open(train_output_file, "w") as f:
        f.writelines(train_set)

    # Write the validation set to a new file
    with open(validation_output_file, "w") as f:
        f.writelines(validation_set)

    client = OpenAI(api_key=api_key)

    with open(train_output_file, "rb") as training_file:
        training_response = client.files.create(file=training_file, purpose="fine-tune")
    training_file_id = training_response.id

    with open(validation_output_file, "rb") as validation_file:
        validation_response = client.files.create(
            file=validation_file, purpose="fine-tune"
        )
    validation_file_id = validation_response.id

    # suffix_name = "codecralers-test"
    response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model="gpt-3.5-turbo",
        suffix=suffix_name,
    )

    # Access the job ID using the appropriate attribute
    job_id = response.id

    response = client.fine_tuning.jobs.retrieve(job_id)

    response = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=50)
    events = response.data
    events.reverse()

    response = client.fine_tuning.jobs.retrieve(job_id)
    fine_tuned_model_id = response.fine_tuned_model

    print("\nFine-tuned model id:", fine_tuned_model_id)

    return fine_tuned_model_id


def fine_tune_model(file_path, suffix_name, api_key):
    dataset = generate_dataset(file_path)
    train_set, validation_set = split_train_validation(input_file)
    fine_tuned_model_id = fine_tuning(
        train_output_file,
        validation_output_file,
        train_set,
        validation_set,
        suffix_name,
        api_key,
    )
    return fine_tuned_model_id
