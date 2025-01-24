SYSTEM_PROMPT = {
    "Qwen/Qwen2-7B-Instruct": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
    "Qwen/QwQ-32B-Preview": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
    "Qwen/Qwen2.5-72B-Instruct": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
    "Qwen/Qwen2.5-32B-Instruct": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
    "Qwen/Qwen2.5-7B-Instruct": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
    "Qwen/Qwen2.5-1.5B-Instruct": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
    "Qwen/Qwen2.5-Math-7B-Instruct": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
    "Qwen/Qwen2.5-Math-72B-Instruct": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
    "PRIME-RL/Eurus-2-7B-PRIME": """When tackling complex reasoning tasks, you have access to the following actions. Use them as needed to progress through your thought process. After each action, determine and state the next most appropriate action to take.

Actions:

{actions}

Your action should contain multiple steps, and each step starts with #. After each action (except OUTPUT), state which action you will take next with ''Next action: [Your action]'' and finish this turn. Continue this process until you reach a satisfactory conclusion or solution to the problem at hand, at which point you should use the [OUTPUT] action. The thought process is completely invisible to user, so [OUTPUT] should be a complete response. You should strictly follow the format below:

[ACTION NAME]

# Your action step 1

# Your action step 2

# Your action step 3

...

Next action: [NEXT ACTION NAME]


Now, begin with the [ASSESS] action for the following task:
""",
    "NovaSky-AI/Sky-T1-32B-Preview": "Your role as an assistant involves thoroughly exploring questions through a systematic long \
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
        Now, try to solve the following question through the above guidelines:",
    "openai/o1-mini": "Question: {input}\nAnswer: ",
    "openai/o1-preview": "Question: {input}\nAnswer: ",
    "openai/gpt-4o-mini": "User: {input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\nAssistant:",
}

MODEL_TO_NAME = {
    "Qwen/Qwen2-7B-Instruct": "Qwen2-7B-Instruct",
    "Qwen/QwQ-32B-Preview": "QwQ-32B-Preview",
    "Qwen/Qwen2.5-72B-Instruct": "Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct": "Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-Math-7B-Instruct": "Qwen2.5-Math-7B-Instruct",
    "Qwen/Qwen2.5-Math-72B-Instruct": "Qwen2.5-Math-72B-Instruct",
    "PRIME-RL/Eurus-2-7B-PRIME": "Eurus-2-7B-PRIME",
    "NovaSky-AI/Sky-T1-32B-Preview": "Sky-T1-32B-Preview",
    "openai/o1-mini": "o1-mini",
    "openai/o1-preview": "o1-preview",  
    "openai/gpt-4o-mini": "gpt-4o-mini",
}

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
"""

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
"""