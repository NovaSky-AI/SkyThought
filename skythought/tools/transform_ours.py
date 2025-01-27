import json
import argparse

def read_json(input_path):
    # Reads a JSON file and returns the data.
    with open(input_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def write_json(data, output_path):
    # Writes data to a JSON file.
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def concat_data(data):
    # Concatenates system prompts and conversations from a list of entries.
    concatenated_texts = []
    for entry in data:
        if 'system' in entry and 'conversations' in entry:
            system_prompt = entry['system']
            conversations = entry['conversations']
            conversation_texts = ' '.join(convo['value'] for convo in conversations if 'from' in convo and 'value' in convo)
            full_text = f"{system_prompt}\n\n{conversation_texts}"
            concatenated_texts.append(full_text)
        else:
            raise ValueError("Missing required keys in JSON entry")
    return concatenated_texts

def main():
    parser = argparse.ArgumentParser(description="Concatenate text from JSON entries and save to a new JSON file.")
    parser.add_argument("--input", help="Input JSON file path")
    parser.add_argument("--output", help="Output JSON file path")
    args = parser.parse_args()

    # Read the original JSON data
    input_data = read_json(args.input)
    
    # Concatenate the required texts
    concatenated_texts = concat_data(input_data)
    
    # Prepare the new JSON data as a list of text entries
    output_data = [{"text": text} for text in concatenated_texts]
    
    # Write the new JSON data to a file
    write_json(output_data, args.output)

if __name__ == "__main__":
    main()