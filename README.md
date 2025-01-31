## Additional Comments

Reference code for xIELU (pronounced like "shear lu") and xIPReLU (prounounced like "zip relu"), based on the nanotron codebase https://github.com/huggingface/nanotron

/src/nanotron/nn/activations.py - contains the code for xIELU and xIPReLU

/src/nanotron/models/llama.py - contains the code for Llama transformer architecture

/configs/ - contains pretraining configs for 1.1B and 3B models on 125B tokens using WSD

/xielu-cuda/ - A kernel fusion implementation for xIELU is under development. Currently achieves 1.75x speedup compared to the PyTorch implementation relying on torch.compile(), but needs additional modifications to support half/bf16 precisions.

## Fused xIELU Benchmark Results

| Model | Batch | SeqLen | HiddenDim | Fwd (ms) | Bwd (ms) | Speedup |
|-------|--------|---------|------------|-----------|-----------|---------|
| XIELU-Python | 4 | 16 | 128 | 0.06 | 0.15 | - |
| XIELU-Cuda | 4 | 16 | 128 | 0.04 | 0.09 | 1.63x |
| XIELU-Python | 8 | 32 | 256 | 0.07 | 0.19 | - |
| XIELU-Cuda | 8 | 32 | 256 | 0.04 | 0.09 | 1.85x |
| XIELU-Python | 16 | 32 | 128 | 0.07 | 0.19 | - |
| XIELU-Cuda | 16 | 32 | 128 | 0.04 | 0.10 | 1.73x |
| XIELU-Python | 16 | 64 | 512 | 0.06 | 0.27 | - |
| XIELU-Cuda | 16 | 64 | 512 | 0.06 | 0.14 | 1.70x |
| XIELU-Python | 32 | 128 | 1024 | 0.09 | 0.36 | - |
| XIELU-Cuda | 32 | 128 | 1024 | 0.07 | 0.15 | 2.00x |
| XIELU-Python | 5 | 4096 | 8192 | 0.54 | 1.91 | - |
| XIELU-Cuda | 5 | 4096 | 8192 | 0.33 | 1.07 | 1.75x |
| XIELU-Python | 50 | 4096 | 8192 | 4.89 | 18.96 | - |
| XIELU-Cuda | 50 | 4096 | 8192 | 3.35 | 10.66 | 1.76x |

The speedup is calculated based on the total time (Fwd + Bwd) of CUDA implementation compared to PyTorch for each configuration.

<h1 align="center">⚡️ Nanotron</h1>

<p align="center">
    <a href="https://github.com/huggingface/nanotron/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/nanotron.svg">
    </a>
    <a href="https://github.com/huggingface/nanotron/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/huggingface/nanotron.svg?color=green">
    </a>
</p>

<h4 align="center">
    <p>
        <a href="#installation">Installation</a> •
        <a href="#quick-start">Quick Start</a> •
        <a href="#features">Features</a> •
        <a href="CONTRIBUTING.md">Contributing</a>
    <p>
</h4>

<h3 align="center">
    <a href="https://huggingface.co/nanotron"><img style="float: middle; padding: 10px 10px 10px 10px;" width="60" height="55" src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png" /></a>
</h3>
<h3 align="center">
<p>Pretraining models made easy
</h3>


Nanotron is a library for pretraining transformer models. It provides a simple and flexible API to pretrain models on custom datasets. Nanotron is designed to be easy to use, fast, and scalable. It is built with the following principles in mind:

- **Simplicity**: Nanotron is designed to be easy to use. It provides a simple and flexible API to pretrain models on custom datasets.
- **Performance**: Optimized for speed and scalability, Nanotron uses the latest techniques to train models faster and more efficiently.

## Installation

```bash
# Requirements: Python>=3.10
git clone https://github.com/huggingface/nanotron
cd nanotron
pip install --upgrade pip
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
pip install -e .

# Install dependencies if you want to use the example scripts
pip install datasets transformers
pip install triton "flash-attn>=2.5.0" --no-build-isolation
```
> [!NOTE]
> If you get `undefined symbol: ncclCommRegister` error you should install torch 2.1.2 instead: `pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121`

> [!TIP]
> We log to wandb automatically if it's installed. For that you can use `pip install wandb`. If you don't want to use wandb, you can run `wandb disabled`.

## Quick Start
### Training a tiny Llama model
The following command will train a tiny Llama model on a single node with 8 GPUs. The model will be saved in the `checkpoints` directory as specified in the config file.
```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=8 run_train.py --config-file examples/config_tiny_llama.yaml
```

### Run generation from your checkpoint
```bash
torchrun --nproc_per_node=1 run_generate.py --ckpt-path checkpoints/10/ --tp 1 --pp 1
# We could set a larger TP for faster generation, and a larger PP in case of very large models.
```

### Custom examples
You can find more examples in the [`/examples`](/examples) directory:
<!-- Make a table of the examples we support -->
| Example | Description |
| --- | --- |
| `custom-dataloader` | Plug a custom dataloader to nanotron |
| `datatrove` | Use the datatrove library to load data |
| `doremi` | Use DoReMi to speed up training |
| `mamba` | Train an example Mamba model |
| `moe` | Train an example Mixture-of-Experts (MoE) model |
| `mup` | Use spectral µTransfer to scale up your model |

We're working on adding more examples soon! Feel free to add a PR to add your own example. 🚀


## Features
We currently support the following features:
- [x] 3D parallelism (DP+TP+PP)
- [x] Expert parallelism for MoEs
- [x] AFAB and 1F1B schedules for PP
- [x] Explicit APIs for TP and PP which enables easy debugging
- [x] ZeRO-1 optimizer
- [x] FP32 gradient accumulation
- [x] Parameter tying/sharding
- [x] Custom module checkpointing for large models
- [x] Spectral µTransfer parametrization for scaling up neural networks
- [x] Mamba example

And we have on our roadmap:
- [ ] FP8 training
- [ ] ZeRO-3 optimizer (a.k.a FSDP)
- [ ] `torch.compile` support
- [ ] Ring attention
- [ ] Interleaved 1f1b schedule

## Credits
We would like to thank everyone working on LLMs, especially those sharing their work openly from which we took great inspiration: Nvidia for `Megatron-LM/apex`, Microsoft for `DeepSpeed`, HazyResearch for `flash-attn`..
