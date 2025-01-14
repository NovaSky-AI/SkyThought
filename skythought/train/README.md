## Training
We use a fork from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to perform training.

Step 1: Please add the data path produced by the tools directory or the one we provide, to the file_name field of Sky-T1 entry in [LLaMA-Factory/data/dataset_info.json](./LLaMA-Factory/data/dataset_info.json).

Step 2: run 

`FORCE_TORCHRUN=1 NNODES=1 NODE_RANK=0 MASTER_PORT=29501 llamafactory-cli train examples/train_full/qwen2_full_sft.yaml`

 to train from a 32B model on 8 H100 GPUs. Interested readers can refer to the detailed settings in [examples/train_full/qwen2_full_sft.yaml](./LLaMA-Factory/examples/train_full/qwen2_full_sft.yaml).
