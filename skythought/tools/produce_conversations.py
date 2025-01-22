import os
import json
import argparse
import random

def main():
    parser = argparse.ArgumentParser(description="Filter and match conversations based on correctness.")
    parser.add_argument("--correctness_file", type=str, required=True, help="File containing correctness data")
    parser.add_argument("--match_file", type=str, required=True, help="File to match and filter")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of random samples to select from the filtered data")
    parser.add_argument("--store_wrong", action="store_true", help="Store wrong conversations")


    args = parser.parse_args()

    # Load correctness data
    with open(args.correctness_file, 'r') as f:
        correctness_data = json.load(f)

    # Load file to match
    with open(args.match_file, 'r') as f:
        match_data = json.load(f)
    # Initialize filtered data list
    filtered_data = []
    total_not_found = 0

    # Iterate through match data
    for data in match_data:
        cur_key = data["conversations"][0]["value"]
        found_match = False

        # Search for matching key in correctness data
        for key, value in correctness_data.items():
            if key in cur_key:
                # Check correctness if key is found
                correctness = value["responses"]["0"]["correctness"]
                # continue 
                if correctness is None:
                    print(f"Not selected: {correctness}")
                    break
                if args.store_wrong:
                    if correctness:
                        print(f"Not selected: {correctness}")
                        break
                    else:
                        print(f"Selected: {correctness}")
                        data["conversations"][0]["value"] = "Return your final response within \\boxed{{}}. " + data["conversations"][0]["value"]
                        filtered_data.append(data)
                else:
                    if not correctness:
                        print(f"Not selected: {correctness}")
                        break
                    else:
                        print(f"Selected: {correctness}")
                        data["conversations"][0]["value"] = "Return your final response within \\boxed{{}}. " + data["conversations"][0]["value"]
                        filtered_data.append(data)
                found_match = True
                break

        if not found_match:
            print(f"Warning: No matching key found for content: {cur_key[:100]}...")
            total_not_found += 1
    # Save filtered data
   # with open(args.output, 'w') as f:
   #     json.dump(filtered_data, f, indent=4, ensure_ascii=False)

    print(f"Filtered {len(filtered_data)} conversations from {len(match_data)} total.")
    if args.num_samples > 0:
        sampled_data = random.sample(filtered_data, args.num_samples)
        with open(args.output, 'w') as f:
            json.dump(sampled_data, f, indent=4, ensure_ascii=False)

    print(f"Filtered {len(sampled_data)} conversations from {len(match_data)} total after random sampling.")
    print(f"Output saved to {args.output}")
    print(f"Total not found: {total_not_found}")
if __name__ == "__main__":
    main()
