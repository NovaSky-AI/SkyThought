import json
import random

from .prompts import system_prompt

still2_jsonl_file = "../../data/public_long_form_thought_data_5k.jsonl"
code_json_file = "../../data/converted_apps_long_form_thought_data_5k.json"
output_file = "../../data/converted_v2_long_form_thought_data_9k.json"

# Load the JSONL file
still2_data = []
with open(still2_jsonl_file, "r") as f:
    for line in f:
        still2_data.append(json.loads(line.strip()))
# print(math_data)

# Process the data into the desired format
all_data = []
code_num = 0

for entry in still2_data:
    question = entry["question"]
    combined_text = entry["combined_text"]
    domain = entry["domain"]
    if domain != "code":
        # Create the conversation format
        conversations = [
            {"from": "user", "value": question},
            {"from": "assistant", "value": combined_text},
        ]

        # Prepare the final structure
        cur_data = {"system": system_prompt, "conversations": conversations}
        all_data.append(cur_data)
    else:
        code_num += 1

print(code_num)
with open(code_json_file, "r") as f:
    code_data = json.load(f)
# print(code_data[0])

all_data.extend(code_data)
print(
    f"First item slice before shuffle: {all_data[0]['conversations'][-1]['value'][-50:-1]}"
)
random.shuffle(all_data)
print(
    f"First item slice after shuffle: {all_data[0]['conversations'][-1]['value'][-50:-1]}"
)
print(len(all_data))

# Save the converted data to the output file
with open(output_file, "w") as f:
    json.dump(all_data, f, indent=4)

print(
    f"Conversion completed. The data has been saved to {output_file} with {len(all_data)} data."
)
