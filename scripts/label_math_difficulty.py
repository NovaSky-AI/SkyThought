import argparse
import ast
import json
import multiprocessing as mp
import os
import re
import time
from itertools import cycle

import openai
from datasets import load_dataset
from tqdm import tqdm

from .prompts import aops_criteria, grading_prompt


# Function to set the OpenAI API key
def set_openai_key(api_key):
    openai.api_key = api_key


# From FastChat
def find_difficulty(judgment):
    one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
    one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")
    match = re.search(one_score_pattern, judgment)
    if not match:
        match = re.search(one_score_pattern_backup, judgment)

    if match:
        rating = ast.literal_eval(match.groups()[0])
    else:
        rating = -1

    return rating


# GPT API processing function with retry logic
def process_content(problem, api_key):
    # Set the OpenAI key for this request
    set_openai_key(api_key)

    # GPT prompt
    prompt = grading_prompt.format(problem=problem, aops_criteria=aops_criteria)
    retries = 3
    while retries > 0:
        try:
            # OpenAI API call
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a math problem difficulty labeler.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2048,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except openai.RateLimitError:
            retries -= 1
            if retries == 0:
                return "Error: Rate limit reached and retries exhausted."
            print("Sleep for 5 seconds for API limit.")
            time.sleep(5)
        except Exception as e:
            return f"Error processing content: {e}"


def process_entry(entry, api_key_cycle):
    # Get the next API key from the cycle
    api_key = next(api_key_cycle)

    # Pass only entry["problem"] to the process_content function
    processed = process_content(entry["problem"], api_key)

    # Store the processed content in the responses
    entry["messages"] = ""
    entry["gpt_difficulty"] = processed
    entry["gpt_difficulty_parsed"] = find_difficulty(processed)
    return entry


# Wrapper function for multiprocessing
def process_entry_wrapper(args):
    return process_entry(*args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label difficulty")
    parser.add_argument("--source", type=str, help="")
    parser.add_argument("--start", type=int, default=0, help="")
    parser.add_argument("--end", type=int, default=-1, help="")
    parser.add_argument(
        "--keys", type=str, help="File containing OpenAI API keys (one per line)."
    )
    args = parser.parse_args()

    dataset = load_dataset("AI-MO/NuminaMath-CoT")
    data = (
        dataset["train"]
        .to_pandas()
        .query("source == @args.source")
        .iloc[args.start : args.end]
    )

    data = data.to_dict(orient="records")

    # Load API keys and prepare a round-robin cycle
    with open(args.keys, "r") as f:
        api_keys = [line.strip() for line in f if line.strip()]
    api_key_cycle = cycle(api_keys)

    # Prepare output file
    output_file = f"labeled_{args.source}_{args.start}_{args.end}.json"

    # Use multiprocessing to process the content
    results = []
    with mp.Pool(os.cpu_count()) as pool:
        tasks = [(entry, api_key_cycle) for entry in data]
        for result in tqdm(pool.imap(process_entry_wrapper, tasks), total=len(data)):
            results.append(result)

    # Aggregate and write results in the main process
    # aggregated_data = {key: values for key, values in results}
    # print(results)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Processed data saved to {output_file}")
