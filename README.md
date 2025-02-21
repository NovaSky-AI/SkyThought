<div align="center">

# SkyThought

[![Github](https://img.shields.io/badge/SkyThought-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/NovaSky-AI/SkyThought) [![Twitter](https://img.shields.io/badge/NovaSky-white?style=for-the-badge&logo=X&logoColor=000&color=000&labelColor=white)](https://x.com/NovaSkyAI) [![Hugging Face Collection](https://img.shields.io/badge/NovaSky-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor)](https://huggingface.co/NovaSky-AI) [![Discord](https://img.shields.io/badge/NovaSky-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/RBAjeWSA)


<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="#news" style="text-decoration: none; font-weight: bold;">News</a> •
    <a href="#links" style="text-decoration: none; font-weight: bold;">Links</a> •
    <a href="#getting-started" style="text-decoration: none; font-weight: bold;">Getting Started</a> •
    <a href="#evaluation" style="text-decoration: none; font-weight: bold;">Evaluation</a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">Citation</a> •
    <a href="#acknowledgement" style="text-decoration: none; font-weight: bold;">Acknowledgement</a> 
  </p>
</div>

</div>


# News
- **[2025/02/21]** 🎉 We released S*: Test time scaling for code generation ([paper](https://arxiv.org/pdf/2502.14382), [code](https://github.com/NovaSky-AI/SkyThought/tree/main/skythought/test-time-scaling)), a simple and extensible test time scaling framework for code generation.
- **[2025/02/11]** 🎉 We released Sky-T1-7B ([model](https://huggingface.co/NovaSky-AI/Sky-T1-7B)) and Sky-T1-mini ([model](https://huggingface.co/NovaSky-AI/Sky-T1-mini)) to demonstrate the potential of RL in further enhancing model's capability beyond distillation.
- **[2025/01/23]** ⚡️ We released Sky-T1-32B-Flash ([model](https://huggingface.co/NovaSky-AI/Sky-T1-32B-Flash), [data](https://huggingface.co/datasets/NovaSky-AI/Sky-T1_preference_data_10k)) to tackle overthinking and reduce reasoning sequence lengths while maintaining accuracy.
- **[2025/01/19]** 🎉 [Chat demo](http://164.152.23.196:3000/) for Sky-T1-32B-Preview is alive! Please check it out!
- **[2025/01/10]** 🎉 We have released our Sky-T1-32B-Preview [model](https://huggingface.co/NovaSky-AI/Sky-T1-32B-Preview) and [data](https://huggingface.co/datasets/NovaSky-AI/Sky-T1_data_17k) through [HuggingFace](https://huggingface.co/NovaSky-AI)!


# Links

- 📜 [Sky-T1-7B and Sky-T1-mini Blog Post](https://novasky-ai.github.io/posts/sky-t1-7B/)
- 📜 [Sky-T1-32B-Flash Blog Post](https://novasky-ai.github.io/posts/reduce-overthinking/)
- 📜 [Sky-T1-32B-Preview model Blog Post](https://novasky-ai.github.io/posts/sky-t1/)
- 🤗 [Sky-T1-32B-Preview model](https://huggingface.co/NovaSky-AI)

# Getting Started

We open source the code and scripts we used for data curation, training, and evaluation for Sky-T1-32B-Preview, you can find more details in each directory.
- [`recipes`](./recipes/): Recipes - data curation steps and training strategies - for building our models `Sky-T1-32B-Flash`, `Sky-T1-32B-Preview` and `Sky-T1-7B` series. 
- [`skythought/evals`](./skythought/evals/): Our data generation and evaluation library. 
- [`skythought/train`](./skythought/train/): Training scripts for Sky-T1. We use [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory) to perform training. 
- [`skythought/skythought-rl`](./skythought/skythought-rl/): RL training code for Sky-T1-7B and Sky-T1-mini.

# Evaluation

## Usage

First, clone the repository and install the package

```shell
git clone https://github.com/NovaSky-AI/SkyThought.git
cd SkyThought
# installs shown for uv
uv venv python==3.10
source .venv/bin/activate
uv pip install -e .
```

Running evaluation is as simple as: 

```bash
skythought evaluate --model NovaSky-AI/Sky-T1-32B-Preview --task aime
```

We support a wide variety of datasets in mathematics, science and coding:

- AIME'24
- MATH500
- GPQADiamond
- MMLU
- ARC-Challenge
- OlympiadBench
- AMC'23 
- TACO 
- APPS
- LiveCodeBench
- MMLU Pro
- MinervaMath
- GSM8K
- AIME'25

For more details, please refer to our [evaluation guide](examples/evaluate.ipynb) and the [README](skythought/evals/README.md).


### Evaluation results
Following, we show our evaluation results for the Sky-T1-32B-Preview model across math, coding, and science benchmarks.

| Metric                | Sky-T1-32B-Preview | Qwen-2.5-32B-Instruct | QwQ   | o1-preview |
|-----------------------|---------------------|--------|-------|------------|
| Math500              | 86.4                    | 81.4    | 92.2 | 81.4       |
| AIME2024             | 43.3                    | 16.7    | 50.0  | 40.0       |
| LiveCodeBench-Easy   | 86.3                    | 84.6   | 90.7  | 92.9       |
| LiveCodeBench-Medium | 56.8                    | 40.8   | 56.3  | 54.9       |
| LiveCodeBench-Hard   | 17.9                    | 9.8   | 17.1  | 16.3       |
| GPQA-Diamond         | 56.8                    | 45.5   | 52.5  | 75.2       |
| OlympiadBench (Math, EN)    | 59.79	           | 46.74	| 62.17	 | 59.2      | 

#### Results on non-reasoning benchmarks

We also evaluate on non-reasoning benchmarks (these are benchmarks for instruction-following, QA, etc) to test whether the model has traded-off capability in other domains for better performance in reasoning-related benchmarks. 


| Metric | Sky-T1-32B-Preview | Qwen-2.5-32B-Instruct | QwQ-32B-Preview | Eval Implementation |
|---------|-------------------|---------------------|-----------------|-------------------|
| MMLU (0 shot; no CoT) | **78.36** | 74.14 | 71.23 | [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness) |
| MMLU (5 shot; no CoT) | 82.46 | **82.62** | 82.32 | [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness) |
| ARC-C (0 shot; no CoT) | **49.49** | 49.4 | 49.66 | [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness) |
| IFEval | 75.79 | **78.74** | 42.51 | [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness) |
| LLM-as-a-Judge | 9.12	| **9.19** | 8.30 | [fastchat](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) |
| MGSM (0 shot; `direct`) | 33 | **42.3** | 19.07 | [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness) |
| MGSM (8-shot; `direct`) | 58.4 | **61.47** | 58.5 | [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness) |
| BFCL-v3 | 53.18 | **58.92** | 17.41 | [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) |
| Arena-Hard | **74.79** | 66.51 | 52.6 | [Arena-Hard-Auto](https://github.com/lmarena/arena-hard-auto) |

For more details, refer [here](./skythought/evals/base_instruct_evals.md).

## Fully Open-source: Driving Progress Together
We believe that open-source collaboration drives progress, and with Sky-T1-32B-Preview, we are fully committed to empowering the community. We open-source all details (i.e., data, codes, model weights) to enable the community to replicate and improve on our results *easily*:

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th style="background-color: #f2f2f2;"><div align="center">Sky-T1-32B-Preview</div></th>
      <th><div align="center">STILL-2</div></th>
      <th><div align="center">Journey</div></th>
      <th><div align="center">QwQ</div></th>
      <th><div align="center">o1</div></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Data</td>
      <td style="background-color: #f2f2f2;"><div align="center">✅</div></td>
      <td><div align="center">✅</div></td>
      <td><div align="center">❌</div></td>
      <td><div align="center">❌</div></td>
      <td><div align="center">❌</div></td>
    </tr>
    <tr>
      <td>Code</td>
      <td style="background-color: #f2f2f2;"><div align="center">✅</div></td>
      <td><div align="center">❌</div></td>
      <td><div align="center">❌</div></td>
      <td><div align="center">❌</div></td>
      <td><div align="center">❌</div></td>
    </tr>
    <tr>
      <td>Report</td>
      <td style="background-color: #f2f2f2;"><div align="center">✅</div></td>
      <td><div align="center">✅</div></td>
      <td><div align="center">✅</div></td>
      <td><div align="center">❌</div></td>
      <td><div align="center">❌</div></td>
    </tr>
    <tr>
      <td>Math domain</td>
      <td style="background-color: #f2f2f2;"><div align="center">✅</div></td>
      <td><div align="center">✅</div></td>
      <td><div align="center">✅</div></td>
      <td><div align="center">✅</div></td>
      <td><div align="center">✅</div></td>
    </tr>
    <tr>
      <td>Coding domain</td>
      <td style="background-color: #f2f2f2;"><div align="center">✅</div></td>
      <td><div align="center">❌</div></td>
      <td><div align="center">❌</div></td>
      <td><div align="center">✅</div></td>
      <td><div align="center">✅</div></td>
    </tr>
    <tr>
      <td>Model Weights</td>
      <td style="background-color: #f2f2f2;"><div align="center">✅</div></td>
      <td><div align="center">✅</div></td>
      <td><div align="center">❌</div></td>
      <td><div align="center">✅</div></td>
      <td><div align="center">❌</div></td>
    </tr>
  </tbody>
</table>

# Citation
The code in this repository is mostly described in the post below. Please consider citing this work if you find the repository helpful. 

```bibtex
@misc{sky_t1_2025,
  author       = {NovaSky Team},
  title        = {Sky-T1: Train your own O1 preview model within $450},
  howpublished = {https://novasky-ai.github.io/posts/sky-t1},
  note         = {Accessed: 2025-01-09},
  year         = {2025}
}
```

# Acknowledgement
This work is done at [Berkeley Sky Computing Lab](https://sky.cs.berkeley.edu/), with the amazing compute support from [Lambda Labs](https://lambdalabs.com/service/gpu-cloud?srsltid=AfmBOop5FnmEFTkavVtdZDsLWvHWNg6peXtat-OXJ9MW5GMNsk756PE5), [Anyscale](https://www.anyscale.com/), and [Databricks](https://www.databricks.com/). We would like to express our gratitude for the valuable academic feedback and support from the [Still-2 Team](https://arxiv.org/pdf/2412.09413), and Junyang Lin from the [Qwen Team](https://qwenlm.github.io/).


