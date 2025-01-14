# -*- coding: utf-8 -*-
"""Multimodal_Reasoning_No_vLLM.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/17B2bvuGHEyiNCU3R3dUXYfzVlwGA3Ga_
"""

# Install necessary libraries
!pip install git+https://github.com/huggingface/transformers
!pip install qwen-vl-utils
!pip install datasets

# Mount Google Drive
from google.colab import drive
import os
import json
import re
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_dataset

drive.mount('/content/drive')

# Create necessary directories in Google Drive
# output_dir = "/content/drive/My Drive/World Models"
os.makedirs(output_dir, exist_ok=True)

# Use GPU
if torch.cuda.is_available():
    print("GPU is available and ready to use!")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available. Make sure it's enabled in the runtime settings.")

# Load model and processor
# model_save_path = os.path.join(output_dir, "qwen_model")
# if os.path.exists(model_save_path):
#     print("Loading model from local save path.")
#     model = Qwen2VLForConditionalGeneration.from_pretrained(
#         model_save_path, torch_dtype="auto", device_map="auto"
#     )
#     processor = AutoProcessor.from_pretrained(model_save_path)
# else:
print("Downloading model. This may take some time...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    # model.save_pretrained(model_save_path)
    # processor.save_pretrained(model_save_path)

import gc

# Load the MathVista dataset
dataset = load_dataset("AI4Math/MathVista")

# Use the testmini subset for evaluation
# subset_dataset = dataset["testmini"]
subset_dataset = dataset["testmini"].select(range(20))
# subset_dataset = dataset["testmini"].select(range(20, 200))

# Prepare results storage
responses = {}

import time
start_time = time.time()

# Generate responses
for idx, example in enumerate(subset_dataset):

    with torch.no_grad():  # Wrap the entire processing
        query = example["query"]  # Use the query field directly from the dataset
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example["decoded_image"]},
                    {"type": "text", "text": query},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        outputs = model.generate(**inputs, max_new_tokens=128)
        response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        response_cleaned = response.split("\nassistant\n")[-1].strip()

    responses[example["pid"]] = {
        "query": query,
        "response": response_cleaned,
        "question_type": example["question_type"],
        "answer_type": example["answer_type"],
        "choices": example.get("choices"),
        "precision": example.get("precision"),
        "answer": example["answer"],
    }
    gc.collect()
    torch.cuda.empty_cache()

time_cost = time.time() - start_time
print(f"Time cost for generating responses: {time_cost:.2f} seconds")

