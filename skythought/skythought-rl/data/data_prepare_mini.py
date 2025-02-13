# python data_prepare.py --input PRIME-RL/Eurus-2-RL-Data --output ./data/math/ --math
import argparse
import os

from datasets import Dataset, load_dataset
from tqdm import tqdm

short_system_prompt = """
Your role as an assistant involves thoroughly exploring questions through a systematic long \
        thinking process before providing the final precise and accurate solutions. This requires \
        engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, \
        backtracing, and iteration to develop well-considered thinking process. \
        Please structure your response into two main sections: Thought and Solution. \
        In the Thought section, detail your reasoning process using the specified format: \
        <|begin_of_thought|> {thought with steps separated with '\n\n'} \
        <|end_of_thought|> \
        Each step should include detailed considerations such as analisying questions, summarizing \
        relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining \
        any errors, and revisiting previous steps. \
        In the Solution section, based on various attempts, explorations, and reflections from the Thought \
        section, systematically present the final solution that you deem correct. The solution should \
        remain a logical, accurate, concise expression style and detail necessary step needed to reach the \
        conclusion, formatted as follows: \
        <|begin_of_solution|> \
        {final formatted, precise, and clear solution} \
        <|end_of_solution|> \
        Now, try to solve the following question through the above guidelines:

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="PRIME-RL/Eurus-2-RL-Data")
    parser.add_argument("--output", type=str, default="dataset/sky-t1")
    parser.add_argument("--sky-sys", action="store_true")
    args = parser.parse_args()

    hard_set = ["numina_amc_aime", "numina_olympiads"]
    input_dataset = load_dataset(args.input)
    still3 = load_dataset("RUC-AIBOX/STILL-3-Preview-RL-Data")
    for split in ["train"]:
        output_dataset = []
        cur_dataset = input_dataset[split]
        idx = -1
        for data_entry in tqdm(cur_dataset):
            if split == "train" and data_entry["data_source"] not in hard_set:
                continue
            if args.sky_sys:
                prompt = [
                    {"role": "system", "content": short_system_prompt},
                    {
                        "role": "user",
                        "content": data_entry["prompt"][1]["content"].replace(
                            "Present the answer in LaTex format: \\boxed{Your answer}",
                            "Return your final response within \\boxed{{}}.",
                        ),
                    },
                ]
            else:
                prompt = [
                    {
                        "role": "user",
                        "content": data_entry["prompt"][1]["content"].replace(
                            "Present the answer in LaTex format: \\boxed{Your answer}",
                            "Please reason step by step and return your final response within \\boxed{{}}.",
                        ),
                    }
                ]
            data_entry["prompt"] = prompt
            output_dataset.append(data_entry)

        cur_dataset = still3[split]
        for data_entry in tqdm(cur_dataset):
            if args.sky_sys:
                prompt = [
                    {"role": "system", "content": short_system_prompt},
                    {
                        "role": "user",
                        "content": data_entry["question"]
                        + "Return your final response within \\boxed{{}}.",
                    },
                ]
            else:
                prompt = [
                    {
                        "role": "user",
                        "content": data_entry["question"]
                        + "Please reason step by step and return your final response within \\boxed{{}}.",
                    }
                ]
            cur_data = {
                "data_source": "still3",
                "prompt": prompt,
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": data_entry["answer"],
                },
                "extra_info": {"split": "dummy", "index": 0},
            }
            output_dataset.append(cur_data)

        print(len(output_dataset))
        output_dataset = Dataset.from_list(output_dataset)

        output_dataset.to_parquet(os.path.join(args.output, f"{split}.parquet"))

    for split in ["validation"]:
        output_dataset = []
        aime_dataset = load_dataset("AI-MO/aimo-validation-aime")["train"].to_pandas()
        aime_dataset = aime_dataset[aime_dataset["url"].str.contains("2024", na=False)]
        aime_dataset = [row.to_dict() for _, row in aime_dataset.iterrows()]
        for data_entry in tqdm(aime_dataset):
            if args.sky_sys:
                prompt = [
                    {"role": "system", "content": short_system_prompt},
                    {
                        "role": "user",
                        "content": data_entry["problem"]
                        + "\nReturn your final response within \\boxed{{}}.",
                    },
                ]
            else:
                prompt = [
                    {
                        "role": "user",
                        "content": data_entry["problem"]
                        + "\nPlease reason step by step and return your final response within \\boxed{{}}.",
                    }
                ]
            cur_data = {
                "data_source": "aime",
                "prompt": prompt,
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": data_entry["answer"],
                },
                "extra_info": {"split": "dummy", "index": 0},
            }
            output_dataset.append(cur_data)

        math500_dataset = load_dataset("qq8933/MATH500")
        for data_entry in tqdm(math500_dataset["test"]):
            if args.sky_sys:
                prompt = [
                    {"role": "system", "content": short_system_prompt},
                    {
                        "role": "user",
                        "content": data_entry["problem"]
                        + "\nReturn your final response within \\boxed{{}}.",
                    },
                ]
            else:
                prompt = [
                    {
                        "role": "user",
                        "content": data_entry["problem"]
                        + "\nPlease reason step by step and return your final response within \\boxed{{}}.",
                    }
                ]
            cur_data = {
                "data_source": "math500",
                "prompt": prompt,
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": data_entry["answer"],
                },
                "extra_info": {"split": "dummy", "index": 0},
            }
            output_dataset.append(cur_data)

        amc_dataset = load_dataset("AI-MO/aimo-validation-amc")["train"].to_pandas()
        amc_dataset = amc_dataset[amc_dataset["url"].str.contains("2023", na=False)]
        amc_dataset = [row.to_dict() for _, row in amc_dataset.iterrows()]
        for data_entry in tqdm(amc_dataset):
            print(data_entry["answer"])
            if args.sky_sys:
                prompt = [
                    {"role": "system", "content": short_system_prompt},
                    {
                        "role": "user",
                        "content": data_entry["problem"]
                        + "\nReturn your final response within \\boxed{{}}.",
                    },
                ]
            else:
                prompt = [
                    {
                        "role": "user",
                        "content": data_entry["problem"]
                        + "\nPlease reason step by step and return your final response within \\boxed{{}}.",
                    }
                ]
            cur_data = {
                "data_source": "amc",
                "prompt": prompt,
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": str(data_entry["answer"]),
                },
                "extra_info": {"split": "dummy", "index": 0},
            }
            output_dataset.append(cur_data)

        olympiads_dataset = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP")
        for data_entry in tqdm(olympiads_dataset["train"]):
            if args.sky_sys:
                prompt = [
                    {"role": "system", "content": short_system_prompt},
                    {
                        "role": "user",
                        "content": data_entry["question"]
                        + "\nReturn your final response within \\boxed{{}}",
                    },
                ]
            else:
                prompt = [
                    {
                        "role": "user",
                        "content": data_entry["question"]
                        + "\nPlease reason step by step and return your final response within \\boxed{{}}.",
                    }
                ]
            cur_data = {
                "data_source": "OlympiadBench",
                "prompt": prompt,
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": data_entry["final_answer"][0],
                },
                "extra_info": {"split": "dummy", "index": 0},
            }
            output_dataset.append(cur_data)

        print(len(output_dataset))
        output_dataset = Dataset.from_list(output_dataset)

        output_dataset.to_parquet(os.path.join(args.output, f"{split}.parquet"))
