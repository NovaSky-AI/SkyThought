import argparse
import json
import multiprocessing as mp
import os
import time
from itertools import cycle

import openai
from tqdm import tqdm

from .prompts import convert_prompt, convert_prompt_example

global args


# Function to set the OpenAI API key
def set_openai_key(api_key):
    openai.api_key = api_key


# GPT API processing function with retry logic
def process_content(content, api_key):
    # Set the OpenAI key for this request
    set_openai_key(api_key)

    # GPT prompt
    prompt = convert_prompt.format(example=convert_prompt_example, content=content)

    retries = 3
    while retries > 0:
        try:
            # OpenAI API call
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a solution format convertor.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=16384,
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


# Function for multiprocessing
def process_entry(entry, api_key_cycle):
    key, values = entry
    content = values["responses"]["0.7"]["content"]

    # Get the next API key from the cycle
    api_key = next(api_key_cycle)

    processed = process_content(content, api_key)
    values["responses"]["0.7"]["processed_content"] = processed

    return key, values


# Wrapper function for multiprocessing
def process_entry_wrapper(args):
    return process_entry(*args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process content and save results.")
    parser.add_argument(
        "--input_dir", type=str, help="Input directory containing JSON files."
    )
    parser.add_argument(
        "--keys", type=str, help="File containing OpenAI API keys (one per line)."
    )

    global args
    args = parser.parse_args()

    # Load API keys and prepare a round-robin cycle
    with open(args.keys, "r") as f:
        api_keys = [line.strip() for line in f if line.strip()]
    api_key_cycle = cycle(api_keys)

    # Process each file in the input directory
    files = os.listdir(args.input_dir)
    for filename in files:
        if filename.endswith("results.json"):
            input_path = os.path.join(args.input_dir, filename)

            # Load the data
            with open(input_path, "r") as f:
                data = json.load(f)

            # Prepare output file. Overwrite the original file
            output_file = input_path

            # Use multiprocessing to process the content
            results = []
            with mp.Pool(os.cpu_count()) as pool:
                tasks = [(entry, api_key_cycle) for entry in data.items()]
                for result in tqdm(
                    pool.imap(process_entry_wrapper, tasks), total=len(data)
                ):
                    results.append(result)

            # Aggregate and write results in the main process
            aggregated_data = {key: values for key, values in results}
            with open(output_file, "w") as f:
                json.dump(aggregated_data, f, indent=4)

            print(f"Processed data saved to {output_file}")
