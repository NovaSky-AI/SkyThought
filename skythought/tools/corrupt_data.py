"""
Example usage: python corrupt_data.py --input_file input.json --output_file output.json --corruption_ratio 0.3
"""
import json
import regex  # Ensure you have the 'regex' module installed. If not, install it using: pip install regex
import random
import copy
import argparse
import sys


def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        Namespace: Parsed arguments containing input_file, output_file, and corruption_ratio.
    """
    parser = argparse.ArgumentParser(description="Corrupt numbers in a JSON file with a specified corruption ratio.")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSON file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the corrupted JSON file."
    )
    parser.add_argument(
        "--corruption_ratio",
        type=float,
        default=1.0,
        help="Probability of corrupting each number (0.0 to 1.0). Default is 1.0 (always corrupt)."
    )
    return parser.parse_args()


def corrupt_numbers(text, corruption_ratio):
    """
    Replaces standalone integers and decimal numbers in the text with random integers between 0 and 10
    based on the specified corruption ratio. Tracks the total number of numbers found, the number of changes made,
    and the total number of characters in the original numbers that have been changed.

    Args:
        text (str): The input text to process.
        corruption_ratio (float): Probability of corrupting each number.

    Returns:
        tuple: A tuple containing the corrupted text, the number of matches found,
               the number of changes made, and the total number of characters changed.
    """
    # Initialize counters
    counts = {'total_found': 0, 'total_attempted': 0, 'changed': 0}
    total_chars_changed = 0

    # Regex pattern to find all standalone integers and decimal numbers
    pattern = r"\b\d+(\.\d+)?\b"

    # Define the replacement function
    def replace_with_random(match):
        nonlocal total_chars_changed
        original_number = match.group()
        counts['total_found'] += 1  # Increment total numbers found

        # Decide whether to corrupt based on the corruption ratio
        if random.random() < corruption_ratio:
            counts['total_attempted'] += 1  # Increment attempted corruptions
            new_number = str(random.randint(0, 10))  # Generate a random integer between 0 and 10

            # Check if the new number is different from the original
            if original_number != new_number:
                counts['changed'] += 1  # Increment changed numbers if different
                total_chars_changed += len(original_number)  # Add the length of the original number

            return new_number  # Replace with the new random number
        else:
            return original_number  # Keep the original number

    # Perform the substitution using the 'regex' module
    corrupted_text = regex.sub(pattern, replace_with_random, text)

    return corrupted_text, counts['total_found'], counts['changed'], total_chars_changed


def main():
    # Parse command-line arguments
    args = parse_arguments()

    input_file = args.input_file
    output_file = args.output_file
    corruption_ratio = args.corruption_ratio

    # Validate corruption_ratio
    if not (0.0 <= corruption_ratio <= 1.0):
        print("Error: --corruption_ratio must be between 0.0 and 1.0")
        sys.exit(1)

    # Load the JSON data
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading input file: {e}")
        sys.exit(1)

    # Initialize variables to accumulate counts across the dataset
    total_numbers_found = 0
    total_numbers_attempted = 0
    total_numbers_changed = 0
    total_chars_changed = 0
    total_text_length = 0
    entries_with_numbers = 0  # To track entries that contain at least one number

    corrupted_data = []

    # Process each entry in the data
    for index, entry in enumerate(data):
        # Deep copy the current entry to avoid modifying the original data
        cur_data = copy.deepcopy(entry)

        # Extract the response text from the conversations
        try:
            # Adjust the indices based on your data structure
            cur_response = entry["conversations"][1]["value"]
        except (IndexError, KeyError) as e:
            print(f"Skipping entry at index {index} due to missing fields: {e}")
            corrupted_data.append(cur_data)
            continue  # Skip to the next entry if the expected structure is not found

        # Accumulate the total text length
        total_text_length += len(cur_response)

        # Corrupt the response and get counts
        corrupted_response, matches, changes, chars_changed = corrupt_numbers(cur_response, corruption_ratio)

        # Update the copied data with the corrupted response
        cur_data["conversations"][1]["value"] = corrupted_response
        corrupted_data.append(cur_data)

        # Accumulate counts
        if matches > 0:
            entries_with_numbers += 1  # Increment if at least one number was found in this entry
        total_numbers_found += matches
        total_numbers_changed += changes
        total_chars_changed += chars_changed

        # Accumulate attempted corruptions
        total_numbers_attempted += counts['total_attempted'] if 'counts' in locals() else int(matches * corruption_ratio)

        # Optional: Print progress every 500 entries
        if (index + 1) % 500 == 0:
            print(f"Processed {index + 1} entries...")

    # Calculate the portion of text changed
    if total_text_length > 0:
        portion_changed = (total_chars_changed / total_text_length) * 100
    else:
        portion_changed = 0.0

    # Print the first entry before and after corruption for verification
    if data:
        print(f"Before corrupt: {data[0]}")
    if corrupted_data:
        print(f"After corrupt: {corrupted_data[0]}")

    # Print the corruption summary
    print(f"\n=== Corruption Summary ===")
    print(f"Total Entries Processed: {len(data)}")
    print(f"Entries with Numbers: {entries_with_numbers}")
    print(f"Total Numbers Found: {total_numbers_found}")
    print(f"Total Numbers Attempted to Corrupt: {total_numbers_attempted}")
    print(f"Total Numbers Changed: {total_numbers_changed}")
    print(f"Total Characters Changed: {total_chars_changed}")
    print(f"Total Text Length: {total_text_length}")
    print(f"Portion of Text Changed: {portion_changed:.2f}%")

    # Save the corrupted data to the specified JSON output file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(corrupted_data, f, indent=4, ensure_ascii=False)
        print(f"\nCorrupted data saved to {output_file}")
    except Exception as e:
        print(f"Error saving output file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
