import json
from datasets import load_dataset

# Step 1: Load the dataset
ds = load_dataset(
    "togethercomputer/RedPajama-Data-V2",
    name="sample",
    trust_remote_code=True
)

# Step 2: Define a mapping function to rename 'raw_content' to 'text'
def map_to_text(example):
    return {"text": example["raw_content"]}

# Step 3: Apply the mapping to the dataset
# This will transform each example to only include the 'text' field
ds_mapped = ds.map(
    map_to_text,
    remove_columns=["doc_id", "meta", "quality_signals", "raw_content"],
    batched=True  # Process one example at a time to manage memory usage
)

# Step 4: Save the transformed dataset to a JSON file
# Using streaming to handle large datasets efficiently
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
output_file = "transformed_pretrain.json"

# Save the 'train' split of the dataset
save_to_json(ds_mapped["train"], output_file)

print(f"Dataset has been successfully saved to {output_file}")
