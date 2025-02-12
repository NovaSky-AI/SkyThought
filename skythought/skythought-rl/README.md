### Install veRL:
1. Create a conda environment:

```bash
conda create -n verl python==3.9
conda activate verl
pip install -r requirements.txt
```

2. Install common dependencies (required for all backends)

```bash
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip3 install ray

# flash attention 2
pip3 install flash-attn --no-build-isolation
```

3. Install veRL

```bash
pip3 install -e .
```

### Prepare the data
`python data/data_prepare_*.py --output {corresponding path}`

### Launch the training
```bash
cd examples/sky-t1
bash ./run-sky-t1-7b-zero.sh
```


### Acknowledgement
This repo is modified on top of [VeRL](https://github.com/volcengine/verl) and [PRIME](https://github.com/PRIME-RL/PRIME).
