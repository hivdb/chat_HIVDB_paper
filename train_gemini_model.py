from dotenv import load_dotenv
from dotenv import find_dotenv
load_dotenv(find_dotenv())

from google.cloud import aiplatform
import os
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.preview.tuning import sft

import pandas as pd


project_id = os.environ['GEMINI_PROJECT_ID']
region = os.environ['GEMINI_REGION']
BUCKET_NAME = "example"
BUCKET_URI = f"gs://{BUCKET_NAME}"


# see
# https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/tuning/supervised_finetuning_using_gemini.ipynb"


testing_data_path = "gs://github-repo/generative-ai/gemini/tuning/summarization/wikilingua/sft_test_samples.csv"


generation_model = GenerativeModel("gemini-1.0-pro-002")

sft_tuning_job = sft.train(
    source_model="gemini-1.0-pro-002",
    # train_dataset=f"{BUCKET_URI}/sft_train_samples.jsonl",
    train_dataset=testing_data_path,
    # Optional:
    # validation_dataset=f"{BUCKET_URI}/sft_val_samples.jsonl",
    epochs=1,
    learning_rate_multiplier=1,
)

# Get the tuning job info.
sft_tuning_job.to_dict()

