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
- **[2025/01/19]** 🎉 [Chat demo](http://164.152.23.196:3000/) for Sky-T1-32B-Preview is alive! Please check it out!
- **[2025/01/10]** 🎉 We have released our Sky-T1-32B-Preview [model](https://huggingface.co/NovaSky-AI/Sky-T1-32B-Preview) and [data](https://huggingface.co/datasets/NovaSky-AI/Sky-T1_data_17k) through [HuggingFace](https://huggingface.co/NovaSky-AI)!


# Links

- 📜 [Sky-T1-32B-Preview model Blog Post](https://novasky-ai.github.io/posts/sky-t1/)
- 🤗 [Sky-T1-32B-Preview model](https://huggingface.co/NovaSky-AI)

# Getting Started

We open source the code and scripts we used for data curation, training, and evaluation for Sky-T1-32B-Preview, you can find more details in each directory.
- ``/data``: The 17k training data used to train Sky-T1-32B-Preview. We also add the science and riddle portion from the [STILL-2 model](https://arxiv.org/pdf/2412.09413).
- ``skythought/tools``: Training data curation and evaluation for Sky-T1. To generate our training data, we use the QwQ-32B-Preview model. We curate the data mixture to cover diverse domains that require reasoning, and a reject sampling procedure to improve the data quality.
- ``skythought/train``: Training scripts for Sky-T1. We use [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory) to perform training. The model was trained for 3 epochs with a learning rate of 1e-5 and a batch size of 96. Our model training was completed in 19 hours on 8 H100 GPUs using DeepSpeed Zero-3 offloading, costing approximately $450 as per Lambda Cloud pricing. 


# Evaluation
Following, we show our evaluation results for the Sky-T1-32B-Preview model across math, coding, and science benchmarks.

### Evaluation results
| Metric                | Sky-T1-32B-Preview | Qwen-2.5-32B-Instruct | QwQ   | o1-preview |
|-----------------------|---------------------|--------|-------|------------|
| Math500              | 86.4                    | 81.4    | 92.2 | 81.4       |
| AIME2024             | 43.3                    | 16.7    | 50.0  | 40.0       |
| LiveCodeBench-Easy   | 86.3                    | 84.6   | 90.7  | 92.9       |
| LiveCodeBench-Medium | 56.8                    | 40.8   | 56.3  | 54.9       |
| LiveCodeBench-Hard   | 17.9                    | 9.8   | 17.1  | 16.3       |
| GPQA-Diamond         | 56.8                    | 45.5   | 52.5  | 75.2       |



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
This work is done at [Berkeley Sky Computing Lab](https://sky.cs.berkeley.edu/), with the amazing compute support from [Lambda Labs](https://lambdalabs.com/service/gpu-cloud?srsltid=AfmBOop5FnmEFTkavVtdZDsLWvHWNg6peXtat-OXJ9MW5GMNsk756PE5) and [Anyscale](https://www.anyscale.com/). We would like to express our gratitude for the valuable academic feedback and support from the [Still-2 Team](https://arxiv.org/pdf/2412.09413), and Junyang Lin from the [Qwen Team](https://qwenlm.github.io/).


