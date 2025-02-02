import argparse
import json
import os
import random

from skythought_evals.models import ModelConfig
from skythought_evals.util.math_parsing_util import strip_answer_string
from tqdm import tqdm
from vllm import LLM, SamplingParams

SUBPROBLEM_SPLIT_PROMPT = """
  You are given a reasoning sequence that attempts to solve a math problem.
  This sequence contains multiple proposed solutions, then provides a the final solution. 
  Each proposed solution within the sequence follows a different line of thought, usually to double check the answer. 
  Your objective is to identify these separate lines of thought and add the separator string '#####' between the separate lines of thought.
  This is important: Your response should be the original unchanged reasoning sequence, except for '#####' injected into the sequence between distinct lines of thought.
  Do NOT summarize portions of the reasoning sequence with '...'.

  Please keep the sequence that starts with '<|begin_of_solution|>' and ends with '<|end_of_solution|>' as 
  one single sequence with no '#####' inside of the sequence. Add the separator '#####' immediately before '<|begin_of_solution|>'.

  Importantly, only use '#####' if a line of thought presents an answer. 
  If the line of thought does not include an answer, it cannot be considered a separate line of thought, and should not be separated.

  For example, if the input is:
  <|begin_of_thought|>The answer to 2+3 is 5. But wait, let me double check this. 
  If I have two apples and I am given three more apples, I now have 5 apples, so 5 seems like the right answer. 
  Alternatively, 2+3 is the same as 3+2, which is also 5.<|end_of_thought|>
  <|begin_of_solution|>The answer is 5<|end_of_solution|>. 

  Your output should be:
  <|begin_of_thought|>The answer to 2+3 is 5. 
  #####
  But wait, let me double check this. 
  If I have two apples and I am given three more apples, I now have 5 apples, so 5 seems like the right answer.
  ##### 
  Alternatively, 2+3 is the same as 3+2, which is also 5.<|end_of_thought|>
  #####
  <|begin_of_solution|>The answer is 5<|end_of_solution|>. 
"""  # noqa: E501

SUBSOLUTION_EXTRACTION_PROMPT = """
  You are given text of an attemp to solve a math problem. The text contains a final proposed answer to the math problem.

  The text also contains a string '#####' and after this string the ground truth answer is presented.

  Your objective is to determine whether the final proposed answer is equivalent to the ground truth answer.
  The proposed answer and ground truth answer may be in slightly different formats. For example, the proposed answer may be '1/2' but the ground truth is '0.5'.
  Equivalent answers in different formats should be treated as equivalent.
  If the text contains multiple proposed answers, use the final proposed answer.

  You should return only "True" if the proposed answer is equivalent to the ground truth answer and "False" if there is no proposed answer or if the proposed answer is not equivalent to the ground truth.
  Do NOT respond with anything at all except "True" or "False". 
  
  For example, if you are given:
  I believe 2+3 equals 5.
  #####
  The ground truth answer is five.

  Your response should be:
  True

  Another example, if you are given:
  I believe 2+2 equals 4. But wait, it is actually 5.
  #####
  The ground truth answer is five.

  Your response should be:
  True
"""  # noqa: E501


def load_dataset(dataset_path: str):
    data = {}
    with open(dataset_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def make_scoring_conversations(dataset, system_prompt):
    conversations = []
    for _, key in enumerate(dataset):
        problem = dataset[key]
        gt_answer = strip_answer_string(problem["answer"])
        for response_key in problem["responses"]:
            response = problem["responses"][response_key]["content"]
            prompt_text = response + "\n#####\nThe ground truth answer is " + gt_answer
            conversations.append(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text},
                ]
            )

    return conversations


def score_solutions(dataset, responses, outfile):
    idx = 0
    for _, key in tqdm(
        enumerate(dataset), total=len(dataset), desc="Scoring original solutions"
    ):
        problem = dataset[key]
        for response_key in problem["responses"]:
            score = responses[idx].outputs[0].text.strip()
            problem["responses"][response_key]["correctness"] = score == "True"
            idx += 1

    with open(outfile, "w", encoding="utf-8") as new_file:
        json.dump(dataset, new_file, ensure_ascii=False, indent=2)
    return dataset


def filter_solutions(dataset):
    # First filter out incorrect responses.
    for key in dataset:
        problem = dataset[key]
        keys_to_filter = []
        for response_key in problem["responses"]:
            if not problem["responses"][response_key]["correctness"]:
                keys_to_filter.append(response_key)
        for k in keys_to_filter:
            del problem["responses"][k]
            del problem["token_usages"][k]

    # Next, filter out examples with <2 correct responses.
    keys_to_filter = []
    for key in dataset:
        problem = dataset[key]
        if len(problem["responses"]) < 2:
            keys_to_filter.append(key)
    for k in keys_to_filter:
        del dataset[k]

    # Finally, filter for the shortest and longest solutions for each sample.
    for key in dataset:
        problem = dataset[key]
        token_usages = problem["token_usages"]
        shortest_key, shortest_entry = min(
            token_usages.items(), key=lambda x: x[1]["completion_tokens"]
        )
        longest_key, longest_entry = max(
            token_usages.items(), key=lambda x: x[1]["completion_tokens"]
        )
        problem["token_usages"] = {
            "shortest": shortest_entry,
            "longest": longest_entry,
        }
        new_responses = {
            "shortest": problem["responses"][shortest_key],
            "longest": problem["responses"][longest_key],
        }
        problem["responses"] = new_responses

    return dataset


def make_splitting_conversations(data, system_prompt):
    conversations = []
    for problem in data:
        response = data[problem]["responses"]["shortest"]
        prompt_text = response["content"]
        conversations.append(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text},
            ]
        )
    return conversations


def split_solutions(dataset, responses, delimiter):
    outputs = []
    for _, response in tqdm(
        enumerate(responses), total=len(responses), desc="Splitting responses"
    ):
        content = response.outputs[0].text.strip()
        # Split response by configured delimiter.
        split_content = content.split(delimiter)
        split_content = [x.strip() for x in split_content if x != ""]
        outputs.append(split_content)
    for idx, key in enumerate(dataset):
        solutions = outputs[idx]
        problem = dataset[key]
        problem["responses"]["shortest"]["subsolutions"] = solutions
    return dataset


def make_subscoring_conversations(dataset, system_prompt):
    conversations = []
    for _, key in enumerate(dataset):
        problem = dataset[key]
        gt_answer = strip_answer_string(problem["answer"])
        subsolutions = problem["responses"]["shortest"]["subsolutions"]
        for sub in subsolutions:
            prompt_text = sub + "\n#####\nThe ground truth answer is " + gt_answer
            conversations.append(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text},
                ]
            )
    return conversations


def score_subsolutions(dataset, responses):
    idx = 0
    for _, key in tqdm(
        enumerate(dataset), total=len(dataset), desc="Scoring sub-solutions"
    ):
        problem = dataset[key]
        subsolutions = problem["responses"]["shortest"]["subsolutions"]
        scores = []
        for _, _ in enumerate(subsolutions):
            score = responses[idx].outputs[0].text.strip()
            scores.append(score == "True")
            idx += 1
        problem["responses"]["shortest"]["scores"] = scores
    return dataset


def build_response_variants(dataset):
    def clean_response_string(response):
        if "<|end_of_thought|>" not in response:
            response += "<|end_of_thought|>"
        return response

    keys_to_remove = []

    for key, problem in dataset.items():
        scores = problem["responses"]["shortest"]["scores"]
        subsolutions = problem["responses"]["shortest"]["subsolutions"]

        # Check if there are valid scores
        if True not in scores:
            keys_to_remove.append(key)
            continue

        # Build FCS (First Correct Solution)
        fcs_idx = scores.index(True)
        fcs_response = (
            "\n".join(subsolutions[: fcs_idx + 1])
            if fcs_idx < len(scores) - 1
            else "\n".join(subsolutions[:-1])
        )
        fcs_response = clean_response_string(fcs_response) + "\n" + subsolutions[-1]
        problem["responses"]["fcs"] = fcs_response

        # Build FCS + 1
        fcs_plus1_idx = fcs_idx + 1 if fcs_idx + 1 < len(subsolutions) - 1 else fcs_idx
        fcs_plus1_response = "\n".join(subsolutions[: fcs_plus1_idx + 1])
        fcs_plus1_response = (
            clean_response_string(fcs_plus1_response) + "\n" + subsolutions[-1]
        )
        problem["responses"]["fcs_plus1"] = fcs_plus1_response

        # Check if there are valid scores
        if True not in scores[fcs_idx + 1 :]:
            keys_to_remove.append(key)
            continue

        # Build FCS + Reflection
        fcs_reflection_idx = scores.index(True, fcs_idx + 1)
        fcs_reflection_response = (
            "\n".join(subsolutions[: fcs_reflection_idx + 1])
            if fcs_reflection_idx < len(scores) - 1
            else "\n".join(subsolutions[:-1])
        )
        fcs_reflection_response = (
            clean_response_string(fcs_reflection_response) + "\n" + subsolutions[-1]
        )
        problem["responses"]["fcs_reflection"] = fcs_reflection_response

    # Remove problems without valid sub-solutions
    for key in keys_to_remove:
        del dataset[key]

    return dataset


def compute_token_usages(dataset, variants, llm):
    tokenizer = llm.get_tokenizer()
    for key in tqdm(dataset, desc="Computing token usages", total=len(dataset)):
        problem = dataset[key]
        prompt_tokens = problem["token_usages"]["shortest"]["prompt_tokens"]
        for variant in variants:
            problem["token_usages"][variant] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(
                    tokenizer(problem["responses"][variant]).input_ids
                ),
            }
    return dataset


def build_question_prompt(prompt):
    return "Return your final response within \\boxed{{}}" + prompt


def make_preference_conversations(final_dataset, format, system_prompt):
    conversations = []
    for prompt in final_dataset:
        problem = final_dataset[prompt]
        convo = {}
        convo["conversations"] = [
            {
                "from": "system",
                "value": system_prompt,
            },
            {
                "from": "human",
                "value": build_question_prompt(prompt),
            },
        ]
        convo["chosen"] = {
            "from": "gpt",
            "value": problem["responses"][format],
        }
        convo["rejected"] = {
            "from": "gpt",
            "value": problem["responses"]["longest"]["content"],
        }
        conversations.append(convo)

    return conversations


def make_SILC_conversations(dataset, system_prompt):
    keys_to_filter = []
    for prompt in dataset:
        problem = dataset[prompt]
        contition = False
        for response_key in problem["responses"]:
            if not problem["responses"][response_key]["correctness"]:
                wrong_length = problem["token_usages"][response_key][
                    "completion_tokens"
                ]
                for k in problem["responses"]:
                    if (
                        k != response_key
                        and problem["token_usages"][k]["completion_tokens"]
                        > wrong_length
                        and problem["responses"][k]["correctness"]
                    ):
                        contition = True
                        break
                break
        if not contition:
            keys_to_filter.append(prompt)

    for key in keys_to_filter:
        del dataset[key]

    # Build contrastive pairs out of {short incorrect, long correct}
    conversations = []
    for prompt in dataset:
        problem = dataset[prompt]

        shortest_incorrect_key = None
        shortest_incorrect_length = float("inf")

        # Get shortest incorrect.
        for response_key in problem["responses"]:
            if not problem["responses"][response_key]["correctness"]:
                length = problem["token_usages"][response_key]["completion_tokens"]
                if length < shortest_incorrect_length:
                    shortest_incorrect_length = length
                    shortest_incorrect_key = response_key

        # Get next longest correct.
        shortest_correct_longer_key = None
        shortest_correct_longer_length = float("inf")
        for response_key in problem["responses"]:
            if problem["responses"][response_key]["correctness"]:
                length = problem["token_usages"][response_key]["completion_tokens"]
                if (
                    length > shortest_incorrect_length
                    and length < shortest_correct_longer_length
                ):
                    shortest_correct_longer_length = length
                    shortest_correct_longer_key = response_key

        convo = {}
        convo["conversations"] = [
            {
                "from": "system",
                "value": system_prompt,
            },
            {
                "from": "human",
                "value": build_question_prompt(prompt),
            },
        ]
        convo["chosen"] = {
            "from": "gpt",
            "value": problem["responses"][shortest_correct_longer_key]["content"],
        }
        convo["rejected"] = {
            "from": "gpt",
            "value": problem["responses"][shortest_incorrect_key]["content"],
        }
        conversations.append(convo)

    return conversations


def main():
    parser = argparse.ArgumentParser(
        description="Filter, rewrite, and format generated responses for high-quality data curation."
    )
    parser.add_argument(
        "--rewrite-model",
        type=str,
        required=True,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="The model used for response processing.",
    )
    parser.add_argument(
        "--target-model",
        type=str,
        required=True,
        default="NovaSky-AI/Sky-T1-32B-Preview",
        help="The target model the rewritten responses will be used to train.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the starting dataset of generated responses to filter from.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="./",
        help="Result directory to save processed data.",
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Whether to checkpoint the dataset at each step.",
    )
    parser.add_argument(
        "--SILC",
        action="store_true",
        help="Whether to include short-incorrect/long-correct (SILC) preference pairs.",
    )
    parser.add_argument("--tp", type=int, default=8, help="Tensor Parallelism Degree")
    parser.add_argument(
        "--max_tokens", type=int, default=32768, help="Max tokens for the model."
    )
    args = parser.parse_args()

    if args.result_dir and not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # Initialize model for data processing.
    llm = LLM(model=args.rewrite_model, tensor_parallel_size=args.tp)
    sampling_params = SamplingParams(max_tokens=args.max_tokens)

    original_dataset = load_dataset(args.dataset)

    # Filter for the shortest and longest correct solutions.
    filtered_dataset = filter_solutions(original_dataset)
    if args.checkpoint:
        outfile = os.path.join(args.result_dir, "filtered-responses.json")
        with open(outfile, "w", encoding="utf-8") as new_file:
            json.dump(filtered_dataset, new_file, ensure_ascii=False, indent=2)

    # Split the shortest solution into subsolutions using the configured model.
    conversations = make_splitting_conversations(
        filtered_dataset, SUBPROBLEM_SPLIT_PROMPT
    )
    responses = llm.chat(
        messages=conversations, sampling_params=sampling_params, use_tqdm=True
    )
    split_dataset = split_solutions(filtered_dataset, responses, "#####")
    if args.checkpoint:
        outfile = os.path.join(args.result_dir, "split-solutions.json")
        with open(outfile, "w", encoding="utf-8") as new_file:
            json.dump(split_dataset, new_file, ensure_ascii=False, indent=2)

    # Score the subsolutions using the configured model.
    subscoring_conversations = make_subscoring_conversations(
        split_dataset, SUBSOLUTION_EXTRACTION_PROMPT
    )
    responses = llm.chat(
        messages=subscoring_conversations,
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    scored_dataset = score_subsolutions(split_dataset, responses)
    if args.checkpoint:
        outfile = os.path.join(args.result_dir, "scored-subsolutions.json")
        with open(outfile, "w", encoding="utf-8") as new_file:
            json.dump(scored_dataset, new_file, ensure_ascii=False, indent=2)

    # Rewrite response based on variants of combining sub-solutions. Here are examples for
    # FCS, FCS+1, and FCS+Reflection.
    variants_dataset = build_response_variants(scored_dataset)
    if args.checkpoint:
        outfile = os.path.join(args.result_dir, "response-variants.json")
        with open(outfile, "w", encoding="utf-8") as new_file:
            json.dump(variants_dataset, new_file, ensure_ascii=False, indent=2)

    # Add per-variant token counts to dataset for convenience.
    final_dataset = compute_token_usages(
        variants_dataset, ["fcs", "fcs_plus1", "fcs_reflection"], llm
    )

    system_prompt = ModelConfig.from_model_id(args.target_model).system_prompt

    # Generate conversation format for each variant, which can be used in SimPO/DPO/etc.
    fcs_convo = make_preference_conversations(final_dataset, "fcs", system_prompt)
    fcs_plus1_convo = make_preference_conversations(
        final_dataset, "fcs_plus1", system_prompt
    )
    fcs_reflection_convo = make_preference_conversations(
        final_dataset, "fcs_reflection", system_prompt
    )

    # Optionall add short incorrect, long correct (SILC) conversations
    if args.SILC:
        short_incorrect_long_correct_conversations = make_SILC_conversations(
            load_dataset(args.dataset), system_prompt
        )
        for convo in [fcs_convo, fcs_plus1_convo, fcs_reflection_convo]:
            convo += short_incorrect_long_correct_conversations
            random.shuffle(convo)

    # Save final conversation variants.
    fcs_outfile = os.path.join(args.result_dir, "fcs-conversations.json")
    with open(fcs_outfile, "w", encoding="utf-8") as new_file:
        json.dump(fcs_convo, new_file, ensure_ascii=False, indent=2)

    fcs_plus1_outfile = os.path.join(args.result_dir, "fcs_plus1-conversations.json")
    with open(fcs_plus1_outfile, "w", encoding="utf-8") as new_file:
        json.dump(fcs_plus1_convo, new_file, ensure_ascii=False, indent=2)

    fcs_reflection_outfile = os.path.join(
        args.result_dir, "fcs_reflection-conversations.json"
    )
    with open(fcs_reflection_outfile, "w", encoding="utf-8") as new_file:
        json.dump(fcs_reflection_convo, new_file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
