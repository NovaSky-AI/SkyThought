import os, json

def post_process_code(code):
    code = code.split("</code>")[0]
    code = code.replace("```python", "")
    code = code.split("```")[0]
    code = code.replace("<code>", "")
    # print(f"postprocessed code: {code}")
    return code

name_map = {
        "4o-mini": 'openai/gpt-4o-mini',
        "4o": 'openai/gpt-4o',
        "o1-mini": 'openai/o1-mini',
        "o1": 'openai/o1-preview',
        "o3-mini": 'openai/o3-mini',
        "o1-preview": 'openai/o1-preview',
        "qwen7b": 'Qwen/Qwen2.5-Coder-7B-Instruct',
        "qwen32b": 'Qwen/Qwen2.5-Coder-32B-Instruct',
}

if os.path.exists("v4_only_medium_correct_codes.json"):
    ICL_EXAMPLES = json.load(open("v4_only_medium_correct_codes.json", "r"))
else:
    print("No ICL examples available")
    ICL_EXAMPLES = {}