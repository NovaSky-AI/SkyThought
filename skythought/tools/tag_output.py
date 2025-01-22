import json
import argparse
from tqdm import tqdm
import multiprocessing as mp
import openai
from itertools import cycle
import time
import os
from util.prompts import tag_solution_prompt, remove_reflection_prompt
import re
import random

from transformers import AutoModelForCausalLM, AutoTokenizer

global args
# Function to set the OpenAI API key
def set_openai_key(api_key):
    openai.api_key = api_key

# GPT API processing function with retry logic
def process_content(content, api_key):
    # Set the OpenAI key for this request
    set_openai_key(api_key)
    
    # GPT prompt
    prompt = remove_reflection_prompt.format(content=content)
    retries = 3
    while retries > 0:
        try:
            # OpenAI API call
            response = openai.chat.completions.create(
                model="gpt-4o-mini", #"gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful chatbot."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=16384,
                temperature=0.7
            )
            return response.choices[0].message.content
        except openai.RateLimitError:
            retries -= 1
            if retries == 0:
                return "Error: Rate limit reached and retries exhausted."
            print(f"Sleep for 5 seconds for API limit.")
            time.sleep(5)
        except Exception as e:
            return f"Error processing content: {e}"

def remove_revised_content(text):
    """
    Removes all text tagged with [revised content] and its associated content.
    
    Args:
        text (str): Input text containing tagged content.
        
    Returns:
        str: Text with [revised content] sections removed.
    """
    # Use regex to find and remove [revised content] and its text
    cleaned_text = re.sub(r'\[revised\].*?(?=\[|\Z)', '', text, flags=re.DOTALL)
    # Remove any extra spaces or newlines left after removal
    cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text).strip()
    return cleaned_text

# Function for multiprocessing
def process_entry(entry, api_key_cycle, ratio):
    content = entry["conversations"][1]["value"]

    # Decide based on the ratio if a rewrite should occur
    if random.random() > ratio:
        return entry  # Skip rewriting

    # Get the next API key from the cycle
    api_key = next(api_key_cycle)

    # Process content
    processed = process_content(content, api_key)
    entry["conversations"][1]["value"] = processed
    return entry

# Wrapper function for multiprocessing
def process_entry_wrapper(args):
    return process_entry(*args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process content and save results.")
    parser.add_argument("--input_json", type=str, help="Input JSON file.")
    parser.add_argument("--output_json", type=str, help="Output JSON file.")
    parser.add_argument("--keys", type=str, help="File containing OpenAI API keys (one per line).")
    parser.add_argument("--ratio", type=float, default=0.5, help="Probability ratio for performing rewrites (0.0 to 1.0).")

    global args
    args = parser.parse_args()

    # Load API keys and prepare a round-robin cycle
    with open(args.keys, "r") as f:
        api_keys = [line.strip() for line in f if line.strip()]
    api_key_cycle = cycle(api_keys)

    # Load the data
    with open(args.input_json, "r") as f:
        data = json.load(f)

    data = data

    # Initialize counters
    rewrite_count = 0

    # Use multiprocessing to process the content
    results = []
    with mp.Pool(os.cpu_count()) as pool:
        tasks = [(entry, api_key_cycle, args.ratio) for entry in data]
        for task, result in zip(tasks, tqdm(pool.imap(process_entry_wrapper, tasks), total=len(data))):
            original_entry = task[0]
            if result != original_entry:  # Check if rewriting occurred
                rewrite_count += 1
            results.append(result)

    # Aggregate and write results in the main process
    aggregated_data = results
    with open(args.output_json, "w") as f:
        json.dump(aggregated_data, f, indent=4)

    print(f"Processed data saved to {args.output_json}")
    print(f"Total rewrites performed: {rewrite_count}")
