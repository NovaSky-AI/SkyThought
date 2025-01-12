# Downloading Dataset 
Following are the instructions to download our 17k datasets used to train Sky-T1-32B-Preview model.

## HuggingFace
The datasets are available on [HuggingFace](https://huggingface.co/datasets/NovaSky-AI/Sky-T1_data_17k). Download it as follows:
```python
from datasets import load_dataset
ds = load_dataset("NovaSky-AI/Sky-T1_data_17k")
```

## Github LFS
Reference `/data/Sky-T1_data_17k.json` to download data from Github LFS.
