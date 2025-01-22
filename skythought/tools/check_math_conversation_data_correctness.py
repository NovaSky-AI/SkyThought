import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="Process a JSON file and perform checks.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSON file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output JSON file.')
    args = parser.parse_args()

    all_content = {}
    with open(args.input_file, 'r') as file:
        content = json.load(file)

    for c in content:
        # load from ["conversation"][0]["value"]
        # Replace "Return your final response within \\boxed{{}}. " with ""
        first_message = c["conversations"][0]["value"]
        problem = first_message.replace("Return your final response within \\boxed{{}}. ", "")
        cur_json = {
            "problem": problem,
            "responses": {
            "0": {
                "processed_content": c["conversations"][1]["value"],
                "correctness": None,
                }
            }
        }
        all_content[problem] = cur_json

    with open(args.output_file, 'w') as file:
        json.dump(all_content, file, indent=4)

if __name__ == "__main__":
    main()
