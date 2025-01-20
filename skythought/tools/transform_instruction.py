import json
from datasets import load_dataset

# Step 1: Load the dataset
ds = load_dataset(
    "allenai/tulu-3-sft-olmo-2-mixture",
    trust_remote_code=True
)
# Step 2: Define a mapping function to concatenate 'messages' content
def map_to_text(batch):
    texts = []
    for messages in batch["messages"]:
        # if len(messages) != 2:
        #    print(f"Unexpected number of messages: {len(messages)}")
        # Safeguard against unexpected message lengths
        if len(messages) != 2:
            pass
            # concatenated_text = messages[0]["content"] + messages[1]["content"]
        else:
            # Handle cases with fewer than 2 messages
            concatenated_text = "".join([msg["content"] for msg in messages])
            texts.append(concatenated_text)
    return {"text": texts}

# Step 3: Apply the mapping to the dataset with proper batching
ds_mapped = ds.map(
    map_to_text,
    remove_columns=["id", "messages", "source", "dataset"],
    batched=True,          # Enable batch processing
    batch_size=1000,       # Adjust batch size as needed
    num_proc=4             # Utilize multiple processes for faster mapping (optional)
)

# Step 4: Save the transformed dataset to a JSON file
def save_to_json(dataset, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('[')  # Start of JSON list
        first = True
        for example in dataset:
            if not first:
                f.write(',\n')
            else:
                first = False
            json.dump(example, f, ensure_ascii=False)
        f.write(']')  # End of JSON list

# Specify the output file path
output_file = "transformed_instruction.json"

# Save the 'train' split of the dataset
save_to_json(ds_mapped["train"], output_file)

global total_longer
print(f"Final length {ds_mapped}")
print(f"Dataset has been successfully saved to {output_file}")
