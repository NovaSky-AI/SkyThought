import argparse
import json
import os

from .prompts import system_prompt


def main():
    parser = argparse.ArgumentParser(description="Convert JSON data for processing.")
    parser.add_argument(
        "--input_dir", type=str, help="Directory containing result files."
    )
    parser.add_argument("--output", type=str, help="Output JSON file.")
    args = parser.parse_args()

    all_data = []

    # Iterate through all files in the input directory
    for filename in os.listdir(args.input_dir):
        if filename.endswith("results.json"):
            filepath = os.path.join(args.input_dir, filename)
            with open(filepath, "r") as f:
                cur_data = json.load(f)

            for _, v in cur_data.items():
                prompt = v["prompt"]
                response_data = v["responses"]

                for cur_temp, cur_temp_response in response_data.items():
                    # Only support 0.7 for this version
                    assert (
                        cur_temp == "0.7"
                    ), "Only support a single temperature=0.7 now."
                    # Accept this data
                    if cur_temp_response["correctness"]:
                        # Create the conversation format
                        conversations = [
                            {"from": "user", "value": prompt},
                            {
                                "from": "assistant",
                                "value": cur_temp_response["processed_content"],
                            },
                        ]

                        # Prepare the final structure
                        cur_data = {
                            "system": system_prompt,
                            "conversations": conversations,
                        }
                        all_data.append(cur_data)

    # Save the converted data to the output file
    with open(args.output, "w") as f:
        json.dump(all_data, f, indent=4)

    print(
        f"Conversion completed. The data has been saved to {args.output} with {len(all_data)} data."
    )


if __name__ == "__main__":
    main()
