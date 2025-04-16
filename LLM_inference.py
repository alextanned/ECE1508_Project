from time import time
from vllm import LLM, SamplingParams
import json
import random
import numpy as np
from sklearn.metrics import accuracy_score
import argparse
import os
import re

def data_prep(args):
    # Load test data, since we are only doing inference and not training LLM
    with open("test.json", "r", encoding="utf8") as f:
        test_data = json.load(f)
    
    if args.debug:
        test_data = test_data[:100]
    
    # For each sample, determine the correct answer letter.
    y_true = []
    for sample in test_data:
        # Use sorted keys for a consistent ordering of options.
        sorted_keys = sorted(sample['options'].keys())
        correct_index = sorted_keys.index(sample['answer'])
        # Map index (0-4) to letter (A-E).
        letter = chr(ord('A') + correct_index)
        y_true.append(letter)
    
    num_choices = 5
    return test_data, y_true, num_choices

def parse_arguments():
    parser = argparse.ArgumentParser(description="LLM Multiple Choice Inference")
    parser.add_argument(
        '--save-path',
        type=str,
        required=False,
        default="./",
        help='Path where output results will be saved.'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='The name of the LLM model to be used.'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='Debug mode (use a subset of data)'
    )
    parser.add_argument(
        '--cot',
        action='store_true',
        default=False,
        help='use chain of thought prompting'
    )
    parser.add_argument(
        '--fewshot',
        action='store_true',
        default=False,
        help='use fewshot prompting'
    )
    args = parser.parse_args()
    return args

def parse_final_answer(text):
    # This pattern matches various phrases like:
    # "final answer is: A", "The correct answer is E.", "final answer: B", etc.
    pattern = re.compile(
        r'(?:the\s+)?(?:final|correct)\s+answer(?:\s*(?:is)?\s*:?\s*)([A-E])',
        re.IGNORECASE
    )
    match = pattern.search(text)
    if match:
        return match.group(1).upper()
    
    
    # If that doesn't match, try to capture any letter followed by a colon anywhere in the text.
    # This should capture outputs like "B: Peach cobbler dessert..."
    pattern2 = re.compile(r'\b([A-E])\s*:', re.IGNORECASE)
    matches2 = pattern2.findall(text)
    if matches2:
        # Assume the last occurrence is the intended final answer.
        return matches2[-1].upper()
    
    # Fallback: search line by line for a standalone letter
    for line in reversed(text.splitlines()):
        candidate = line.strip().upper().strip('.')
        if candidate in ['A', 'B', 'C', 'D', 'E']:
            return candidate
    return None

def inference(args):
    X_test, y_true, num_choices = data_prep(args)
    
    # Create an LLM.
    llm = LLM(
        model=args.model,
        enable_prefix_caching=True,
        max_model_len=2048,
        disable_sliding_window=True,
    )
    tokenizer = llm.get_tokenizer()
    
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1.0,
        max_tokens=512,
    )
    
    # Define a prompt template for multiple-choice questions.
    prompt_template = (
        "You are a helpful assistant. Given the following question and answer choices, select the correct answer.\n"
        "Question: {query}\n"
        "Answer Choices:\n"
        "A: {A}\n"
        "B: {B}\n"
        "C: {C}\n"
        "D: {D}\n"
        "E: {E}\n"
        "Please answer with the letter corresponding to the correct answer (A, B, C, D, or E). Do not include anything else in your response"
    )

    if args.cot:
        prompt_template = (
            "You are a helpful assistant. Given the following question and answer choices, select the correct answer.\n"
            "Question: {query}\n"
            "Answer Choices:\n"
            "A: {A}\n"
            "B: {B}\n"
            "C: {C}\n"
            "D: {D}\n"
            "E: {E}\n"
            "Please explain your reasoning step by step. After that on a new line, provide only the final answer as a single letter (A, B, C, D, or E)."
        ) 
    elif args.fewshot:
        few_shot_example = (
            "Example:\n"
            "Question: I need to prepare a passionfruit punch bowl, although I don't think I'm able to find actual passionfruits in supermarkets near me?\n"
            "Answer Choices:\n"
            "A: Punch made from orange juice, lemons, ginger ale and passionfruit syrup\n"
            "B: Passionfruit punch made from real passionfruit\n"
            "C: Passionfruit cocktail using real passionfruit\n"
            "D: Passionfruit pudding\n"
            "E: Cranberry punch\n\n"
            "Reasoning: Since the query indicates that fresh passionfruits are hard to find, the option that uses passionfruit syrup (Option A) is most practical.\n"
            "Final Answer: A\n\n"
            "Example:\n"
            "Question: I want to make a sauce that goes with my ham\n"
            "Answer Choices:\n"
            "A: Brown sugar and mustard glaze for ham\n"
            "B: Biscuits with ham, mustard, and horseradish\n"
            "C: Mashed potatoes with ham\n"
            "D: Tomato spaghetti sauce containing sausages and vegetables\n"
            "E: Chocolate and peanut-flavoured sweet sauce for ice cream sundaes\n\n"
            "Reasoning: The query asks for a sauce that pairs well with ham. A brown sugar and mustard glaze (Option A) is a classic and complementary choice.\n"
            "Final Answer: A\n\n"
        )
        prompt_template = (
            "You are a helpful assistant. Given the following question and answer choices, select the correct answer.\n"
            "Question: {query}\n"
            "Answer Choices:\n"
            "A: {A}\n"
            "B: {B}\n"
            "C: {C}\n"
            "D: {D}\n"
            "E: {E}\n"
            "Please explain your reasoning step by step. After that on a new line, provide only the final answer as a single letter (A, B, C, D, or E)."
        ) 
        prompt_template = few_shot_example+prompt_template
    # Create a list of chat-style prompts.
    generating_prompts = []
    for sample in X_test:
        query = sample['query']
        sorted_keys = sorted(sample['options'].keys())
        options = [sample['options'][key] for key in sorted_keys]
        prompt = prompt_template.format(
            query=query,
            A=options[0],
            B=options[1],
            C=options[2],
            D=options[3],
            E=options[4]
        )
        generating_prompts.append([
            {"role": "system", "content": "You are a multiple choice question answering assistant."},
            {"role": "user", "content": prompt}
        ])
    
    # Apply the chat template using the LLM's tokenizer.
    chat_prompts = tokenizer.apply_chat_template(generating_prompts, tokenize=False)
    
    outputs = llm.generate(chat_prompts, sampling_params)
    
    predicted_answers = []
    for i, output in enumerate(outputs):
        if args.cot:
            text = output.outputs[0].text.split("\n")[-1]
            parsed_text = parse_final_answer(text)
            if parsed_text in ['A', 'B', 'C', 'D', 'E']:
                predicted = parsed_text
            else:
                print("----------------------")
                print("No. ", i, "Unexpected response: ", text, "Parsed: ", parsed_text)
                print("full output: ", output.outputs[0].text)
                # print( output.outputs[0].text)
                # If parsing fails, choose a random answer (or handle the error as needed)
                predicted = random.choice(['A', 'B', 'C', 'D', 'E'])
        
        else:

            text = output.outputs[0].text.split("\n")[-1].upper()
            # Parse the predicted answer by looking at the last character (expected to be A-E).
            if text in ['A', 'B', 'C', 'D', 'E']:
                predicted = text
            else:
                print("----------------------")
                print("No. ", i, "Unexpected response: ", text)
                print("full output: ", output.outputs[0].text)
                # print( output.outputs[0].text)
                # If parsing fails, choose a random answer (or handle the error as needed)
                predicted = random.choice(['A', 'B', 'C', 'D', 'E'])
            
        predicted_answers.append(predicted)
    
    acc = accuracy_score(y_true, predicted_answers)
    print("Accuracy:", acc)
    

    with open(os.path.join(args.save_path, "results.txt"), "w") as f:
        f.write("Accuracy: " + str(acc) + "\n")
        for out in outputs:
            f.write(out.outputs[0].text.strip() + "\n")
    
    return outputs, y_true, predicted_answers

if __name__ == "__main__":
    args = parse_arguments()
    outputs, y_true, predicted_answers = inference(args)
