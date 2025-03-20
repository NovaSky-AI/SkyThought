from typing import Any, Dict

STILL2_SYSTEM_PROMPT = "Your role as an assistant involves thoroughly exploring questions through a systematic long \
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
Now, try to solve the following question through the above guidelines:"


def convert_to_sharegpt_format(row: Dict[str, Any], prompt_column, response_column):
    prompt = row[prompt_column]
    # Create the conversation format
    conversations = [
        {"from": "user", "value": prompt},
        {
            "from": "assistant",
            "value": row[response_column],
        },
    ]

    # Prepare the final structure
    cur_data = {
        "system": STILL2_SYSTEM_PROMPT,
        "conversations": conversations,
        # TODO: remove this
        **row,
    }

    return cur_data
