from dotenv import load_dotenv
from dotenv import find_dotenv
load_dotenv(find_dotenv())

from google.cloud import aiplatform
import os
import vertexai
from vertexai.generative_models import GenerativeModel
# from vertexai.preview.tuning import sft
from vertexai.generative_models import GenerationConfig


def list_all_models():
    for model in aiplatform.Model.list():
        print(model.gca_resource.deployed_models[0].endpoint)


def chat(model, content):
    # sft_tuning_job = sft.SupervisedTuningJob(job_name)
    # sft_tuning_job.tuned_model_endpoint_name

    tuned_model = GenerativeModel(
        model.gca_resource.deployed_models[0].endpoint)

    generation_config = GenerationConfig(
        temperature=0,
    )

    result = tuned_model.generate_content(
        content,
        generation_config=generation_config)

    print(result.candidates)


if __name__ == '__main__':
    model = aiplatform.Model(os.environ['GEMINI_MODEL'])
    chat(model, 'whats your name?')
