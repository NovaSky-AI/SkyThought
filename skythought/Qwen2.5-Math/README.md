<a name="readme-top"></a>

<p align="center">
    <img src="https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/assets/logo/qwen2.5_math.png" width="400"/>
<p>

<p align="center">
    <img src="https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen2.5/2024-08-qwen2.5-math-allsize.png" width="800"/>
<p>


<p align="center">
        🤗 <a href="https://huggingface.co/Qwen">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/qwen">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp<a href="https://www.kaggle.com/models/qwen-lm/qwen2-math">Kaggle</a>&nbsp&nbsp  | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen2.5-math/">Blog</a> &nbsp&nbsp ｜ &nbsp&nbsp📖 <a href="https://qwen.readthedocs.io/">Documentation</a>
<br>
<a href="https://github.com/QwenLM/Qwen/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp
</p>


Visit our Hugging Face or ModelScope organization (click the links above). Search checkpoints with names starting with `Qwen2.5-Math-`, and you will find all you need! Enjoy!


## Introduction

A month ago, we released the first series of mathematical LLMs - [Qwen2-Math](https://qwenlm.github.io/blog/qwen2-math/) - of our Qwen family. Today, we have upgraded it and open-sourced **Qwen2.5-Math** series, including base models **Qwen2.5-Math-1.5B/7B/72B**, instruction-tuned models **Qwen2.5-Math-1.5B/7B/72B-Instruct**, and mathematical reward model **Qwen2.5-Math-RM-72B**. 
                                                                              
Unlike Qwen2-Math series which only supports using Chain-of-Thught (CoT) to solve English math problems, Qwen2.5-Math series is expanded to support using both CoT and Tool-integrated Reasoning (TIR) to solve math problems in both Chinese and English. The Qwen2.5-Math series models have achieved significant performance improvements compared to the Qwen2-Math series models on the Chinese and English mathematics benchmarks with CoT. 


Detailed performance and introduction are shown in this <a href="https://qwenlm.github.io/blog/qwen2.5-math/"> 📑 blog</a>.

> <div align="center">
> <b>
> 🚨 Qwen2.5-Math mainly supports solving English and Chinese math problems through CoT and TIR. We do not recommend using this series of models for other tasks.
> </b>
> </div>

## Requirements
* `transformers>=4.37.0` for Qwen2.5-Math models. The latest version is recommended.

> [!Warning]
> <div align="center">
> <b>
> 🚨 This is a must because `transformers` integrated Qwen2 codes since `4.37.0`.
> </b>
> </div>

For requirements on GPU memory and the respective throughput, see similar results of Qwen2 [here](https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html).

## Quick Start

> [!Important]
>
> **Qwen2.5-Math-72B-Instruct** is an instruction model for chatting;
>
> **Qwen2.5-Math-72B** is a base model typically used for few-shot inference, serving as a better starting point for fine-tuning.
> 

### 🤗 Hugging Face Transformers

Qwen2.5-Math can be deployed and inferred in the same way as [Qwen2.5](https://github.com/QwenLM/Qwen2.5). Here we show a code snippet to show you how to use the chat model with `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-Math-72B-Instruct"
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Find the value of $x$ that satisfies the equation $4x+5 = 6x+7$."

# CoT
messages = [
    {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
    {"role": "user", "content": prompt}
]

# TIR
messages = [
    {"role": "system", "content": "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

This time, we also released a mathematical reward model, [Qwen2.5-Math-RM-72B](https://huggingface.co/Qwen/Qwen2.5-Math-RM-72B), based on Qwen2.5-Math-72B-Instruct. Qwen2.5-Math-RM-72B can be easily infered with HuggingFace Transformers with `trust_remote_code`.

> [!Warning]
> 
> We use `temperature=0.7` and `top_p=0.8` for maj@8 and RM@8 sampling with **Qwen2.5-Math-Instruct**, while `temperature=1.0` and `top_p=0.9` with **Qwen2-Math-Instruct**.
> 
```python
import torch
from transformers import AutoModel, AutoTokenizer

model_name = "Qwen/Qwen2.5-Math-RM-72B"
device = "auto" # the device to load the model onto

model = AutoModel.from_pretrained(
    model_name, 
    device_map=device, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

chat = [
    {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
    {"role": "user", "content": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"},
    {"role": "assistant", "content": "To determine how much Janet makes from selling the duck eggs at the farmers' market, we need to follow these steps:\n\n1. Calculate the total number of eggs laid by the ducks each day.\n2. Determine how many eggs Janet eats and bakes for herself each day.\n3. Find out how many eggs are left to be sold.\n4. Calculate the revenue from selling the remaining eggs at $2 per egg.\n\nLet's start with the first step:\n\n1. Janet's ducks lay 16 eggs per day.\n\nNext, we calculate how many eggs Janet eats and bakes for herself each day:\n\n2. Janet eats 3 eggs for breakfast every morning.\n3. Janet bakes 4 eggs for her friends every day.\n\nSo, the total number of eggs Janet eats and bakes for herself each day is:\n\\[ 3 + 4 = 7 \\text{ eggs} \\]\n\nNow, we find out how many eggs are left to be sold:\n\\[ 16 - 7 = 9 \\text{ eggs} \\]\n\nFinally, we calculate the revenue from selling the remaining eggs at $2 per egg:\n\\[ 9 \\times 2 = 18 \\text{ dollars} \\]\n\nTherefore, Janet makes \\boxed{18} dollars every day at the farmers' market."}
] # 3.75

conversation_str = tokenizer.apply_chat_template(
    chat, 
    tokenize=False, 
    add_generation_prompt=False
)

input_ids = tokenizer.encode(
    conversation_str, 
    return_tensors="pt", 
    add_special_tokens=False
).to(model.device)

outputs = model(input_ids=input_ids)
print(outputs[0])

```

### 🤖 ModelScope
We strongly advise users, especially those in mainland China, to use ModelScope. `snapshot_download` can help you solve issues concerning downloading checkpoints.

### Local Demo (Qwen-Agent)

We developed a demo that supports the TIR mode in [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent), which allows running code locally to experience Tool-Integrated Reasoning capabilities of Qwen2.5-Math.

## Performance

### Base Models

We evaluate our Qwen2.5-Math base models on three widely used English math benchmarks GSM8K, Math, and MMLU-STEM. In addition, we also evaluate three Chinese math benchmarks CMATH, GaoKao Math Cloze, and GaoKao Math QA. All evaluations are tested with few-shot chain-of-thought prompting. 

<p align="center">
    <img src="https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen2.5/qwen2.5-math-table.png" width="800"/>
<p>

Compared to Qwen2-Math-1.5B/7B/72B, Qwen2.5-Math-1.5B/7B/72B have achieved significant improvements on all benchmarks. For example, Qwen2.5-Math-1.5B/7B/72B obtains 5.4, 5.0, 6.3 scores improvement on MATH, and 3.4, 12.2, 19.8 scores improvement on GaoKao Math QA.

### Instruction-Tuned Models

We evaluate Qwen2.5-Math-Instruct on mathematical benchmarks in both English and Chinese. In addition to the widely-used benchmarks, such as GSM8K and Math, we also involve more exams that are more challenging to fully inspect the capabilities of Qwen2.5-Math-Instruct, such as OlympiadBench, CollegeMath, GaoKao, AIME2024, and AMC2023. For Chinese mathematical benchmarks, we use CMATH, Gaokao (Chinese College Entrance Examination 2024), and CN Middle School 24 (China High School Entrance Examination 2024).

We report greedy, Maj@8 and RM@8 performance on all benchmarks in the zero-shot setting, except for the multi-choice benchmarks (including MMLU STEM and multiple-choice problems in GaoKao and CN Middle School 24) with a 5-shot setting.

The Qwen2.5-Math-72B-Instruct model outperforms the Qwen2-Math-72B-Instruct model by an average margin of 4.4 and 6.1 points in English and Chinese, respectively, establishing itself as the best open-source mathematical model currently available.

The flagship model, Qwen2.5-Math-72B-Instruct, significantly outperforms both open-source models and leading closed-source models (e.g., GPT-4o, Gemini Math-Specialized 1.5 Pro). Under the TIR setting of RM@8, a high score of 92.9 was achieved on MATH.

With the aid of synthesized pre-training and supervised fine-tuning data from the 72B model, Qwen2.5-Math-7B-Instruct surpasses Qwen2-Math-Instruct 72B in performance. Under CoT and TIR settings, it achieves MATH scores of 83.6 and 85.3, respectively. 

Even our smallest 1.5B model, achieves a MATH score of around 80 when utilizing the Python Interpreter, outperforming the majority of current models in this domain.

<p align="center">
    <img src="https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen2.5/math_instruct_en.jpg#center" width="800"/>
<p>

<p align="center">
    <img src="https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen2.5/math_instruct_zh.jpg#center" width="800"/>
<p>

In more complex mathematical competition evaluations such as AIME 2024 and AMC 2023, Qwen2.5-Math-Instruct also performs well across various settings, including Greedy, Maj@64, RM@64, and RM@256.

With the support of the Qwen2.5-Math-RM-72B, Qwen2.5-Math-1.5B-Instruct, using the RM@256 in CoT mode, successfully solves 29 out of 40 problems on AMC 2023.

Moreover, Qwen2.5-Math-72B-Instruct nearly achieves a perfect score in TIR mode, solving almost all the problems.

On the extremely difficult AIME 2024 benchmark, Claude3 Opus, GPT-4 Turbo, and Gemini 1.5 Pro manage to solve only 1 or 2 questions out of 30.

In contrast, Qwen2.5-Math-72B-Instruct solves 9 problems in Greedy decoding CoT mode and 12 problems in TIR mode. With the help of the RM, Qwen2.5-Math-7B-Instruct could even solve up to 21 problems, further demonstrating the outstanding mathematical problem-solving ability of Qwen2.5-Math-Instruct.

<p align="center">
    <img src="https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen2.5/math_instruct_aime.jpg" width="500"/>
<p>

## Evaluation

Our evaluation is adapted from [math-evaluation-harness](https://github.com/ZubinGou/math-evaluation-harness).
Feel free to reproduce the results of all instruction models in the Qwen2.5-Math series with scripts in [evaluation](./evaluation).

### Requirements

Before the evaluation, please install the required packages with the following command:

```bash
cd latex2sympy
pip install -e .
cd ..
pip install -r requirements.txt 
pip install vllm==0.5.1 --no-build-isolation
pip install transformers=4.42.3
```

Strictly following the versions of requirements is essential to reproduce the reported scores.

### Run

Evaluate Qwen2.5-Math-Instruct series model with the following command:

```bash
PROMPT_TYPE="qwen25-math-cot"

# Qwen2.5-Math-1.5B-Instruct
export CUDA_VISIBLE_DEVICES="0"
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-1.5B-Instruct"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

# Qwen2.5-Math-7B-Instruct
export CUDA_VISIBLE_DEVICES="0"
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-7B-Instruct"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

# Qwen2.5-Math-72B-Instruct
export CUDA_VISIBLE_DEVICES="0,1,2,3"
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-72B-Instruct"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH
```

## Citation
If you find our work helpful, feel free to give us a citation.

```bibtex
@article{yang2024qwen2,
  title={Qwen2 technical report},
  author={Yang, An and Yang, Baosong and Hui, Binyuan and Zheng, Bo and Yu, Bowen and Zhou, Chang and Li, Chengpeng and Li, Chengyuan and Liu, Dayiheng and Huang, Fei and others},
  journal={arXiv preprint arXiv:2407.10671},
  year={2024}
}
```

## Contact Us
If you are interested in leaving a message to either our research team or product team, join our [Discord](https://discord.gg/z3GAxXZ9Ce) or [WeChat groups](https://github.com/QwenLM/Qwen/blob/main/assets/wechat.png)!

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>
