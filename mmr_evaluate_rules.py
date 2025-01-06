# -*- coding: utf-8 -*-
"""MMR_Evaluate_Rules.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1QzpS4Haovj5VyDERDKNsc_RwsujyALU7
"""

from google.colab import drive
drive.mount('/content/drive')

import json
import re

# Process the file
file_path = '/content/drive/My Drive/World Models/responses.json'
def text_to_number(text):
    """
    Convert textual representation of numbers to numerical format.
    E.g., "Zero" -> 0, "One" -> 1
    """
    text = text.strip().lower()
    number_map = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10
    }
    return number_map.get(text, None)

def evaluate_response_fixed(problem):
    """
    Enhanced evaluation for multi-choice and other types of responses.
    Properly handles cases where response contains option letters (e.g., "A") or option content.
    """
    response = problem.get('response', '').strip()
    answer = problem.get('answer', '').strip()
    answer_type = problem.get('answer_type', '')
    choices = problem.get('choices', None)

    # Handle missing fields
    if not response or not answer or not answer_type:
        return False

    try:
        # Handle multiple-choice questions
        if problem.get('question_type') == 'multi_choice':
            # Extract the selected option from response
            match = re.search(r'Answer:?[ \(]?([A-Z])[\)]? ?.*', response, re.IGNORECASE)
            selected_option = match.group(1).upper() if match else response.upper()

            if choices:
                # Check if response is a valid option letter
                if selected_option in [chr(65 + i) for i in range(len(choices))]:
                    selected_answer = choices[ord(selected_option) - 65]
                    # Normalize and compare with the expected answer
                    if selected_answer.strip().lower() == answer.lower():
                        return True

                # Directly compare response to answer (if response is option content)
                if response.strip().lower() == answer.lower():
                    return True

                # Check if both response and answer contain "yes" but not "no", or contain "no" but not "yes"
                response_lower = response.lower()
                answer_lower = answer.lower()
                if ("yes" in response_lower and "yes" in answer_lower and "no" not in response_lower and "no" not in answer_lower) or \
                   ("no" in response_lower and "no" in answer_lower and "yes" not in response_lower and "yes" not in answer_lower):
                    return True

        # Handle free-form integer answers
        if answer_type == 'integer':
            try:
                response_num = text_to_number(response) if not response.isdigit() else int(response)
                answer_num = text_to_number(answer) if not answer.isdigit() else int(answer)
                return response_num == answer_num
            except ValueError:
                return False

        # Handle free-form float answers
        if answer_type == 'float':
            try:
                precision = problem.get('precision', 1e-3)
                return abs(float(response) - float(answer)) < precision
            except ValueError:
                return False

        # Handle text answers
        if answer_type == 'text':
            # Normalize and compare text answers
            return response.lower().strip() == answer.lower().strip()
    except Exception as e:
        # Return False if any parsing error occurs
        return False

    return False


def process_responses_fixed(file_path):
    """
    Process the responses in the given JSON file with enhanced evaluation logic.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    total = len(data)
    correct = 0
    detailed_results = []

    for pid, problem in data.items():
        is_correct = evaluate_response_fixed(problem)
        if is_correct:
            correct += 1
        detailed_results.append({
            'pid': pid,
            'query': problem.get('query', ''),
            'response': problem.get('response', ''),
            'answer': problem.get('answer', ''),
            'is_correct': is_correct
        })

    accuracy = correct / total if total > 0 else 0

    return {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'detailed_results': detailed_results
    }

# Re-run evaluation with fixed logic
fixed_results = process_responses_fixed(file_path)

# Output fixed results
import pprint
# pprint.pprint(fixed_results['accuracy'])  # Print accuracy


# Save detailed results to a file for further analysis
with open('/content/drive/My Drive/World Models/detailed_results.json', 'w') as f:
    json.dump(fixed_results['detailed_results'], f, indent=2)

print(f"Total questions: {fixed_results['total']}")
print(f"Correct answers: {fixed_results['correct']}")
print(f"Accuracy: {fixed_results['accuracy']:.2%}")
