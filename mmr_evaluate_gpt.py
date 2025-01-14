# -*- coding: utf-8 -*-
"""MMR_Evaluate_GPT.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1035f_bgGtQF-grSMURJfifZiu2Pm-GBA
"""

from google.colab import drive
drive.mount('/content/drive')

import json
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key="xxxxxxx")

# Process the file
file_path = '/content/drive/My Drive/World Models/responses.json'

def evaluate_with_gpt(problem):
    """
    Use GPT-4 API to evaluate the correctness of the model's response.
    """
    query = problem.get('query', '').strip()
    response = problem.get('response', '').strip()
    answer = problem.get('answer', '').strip()
    choices = problem.get('choices', None)

    # Construct the prompt
    prompt = f"""
    Question: {query}
    Model's response: {response}
    Expected answer: {answer}
    """
    if choices:
        prompt += f"Choices: {', '.join(choices)}\n"
    prompt += "Is the model's response correct? Please answer 'Yes' or 'No'."

    try:
        # Send the prompt to GPT-4
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            store=True,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Extract the GPT-4 response
        gpt_response = completion.choices[0].message.content.strip()
        is_correct = gpt_response.lower() == 'yes'
        return is_correct, gpt_response
    except Exception as e:
        print(f"Error evaluating problem {problem.get('pid', 'unknown')}: {e}")
        return False, "Error"

def process_responses_with_gpt(file_path):
    """
    Process the responses in the given JSON file using GPT-4 for evaluation.
    Process all available samples.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    total = len(data)
    correct = 0
    detailed_results = []

    for pid, problem in data.items():
        is_correct, gpt_response = evaluate_with_gpt(problem)
        if is_correct:
            correct += 1
        detailed_results.append({
            'pid': pid,
            'query': problem.get('query', ''),
            'response': problem.get('response', ''),
            'answer': problem.get('answer', ''),
            'gpt_response': gpt_response,
            'is_correct': is_correct
        })

    accuracy = correct / total if total > 0 else 0

    return {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'detailed_results': detailed_results
    }

# Re-run evaluation with GPT-4
results_with_gpt = process_responses_with_gpt(file_path)

# Output results
import pprint
pprint.pprint(results_with_gpt['accuracy'])  # Print accuracy

# Save detailed results to a file for further analysis
with open('/content/drive/My Drive/World Models/detailed_results_gpt.json', 'w') as f:
    json.dump(results_with_gpt['detailed_results'], f, indent=2)

