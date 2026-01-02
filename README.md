# AureliusGPT-Torch

## Overview

AureliusGPT-Torch is an 845k, PyTorch and SentencePiece boosted SLM pretrained on _Meditations_ by Marcus Aurelius and other adjacent philosophical works. It is a larger size, more comprehensive reimplementation of 
[AureliusGPT](https://github.com/tarush-ai/AureliusGPT), a smaller model trained on the same core corpus, but using a handwritten/NumPy first (zero PyTorch or prewritten tokenizer) approach.

Rather than reimplementing a custom BPE algorithm and tokenizer backpropogation from scratch like its brother repo, AureliusGPT-Torch trains SentencePiece on its corpus and relies on PyTorch for autograd.

## Data
The original corpus of _Meditations_ by Marcus Aurelius is 89k tokens approximately when tokenized by a SentencePiece BPE tokenizer trained on a vocabulary length of 2,000. Using Kaplan et al. Chinchilla scaling laws,
the expected parameter size of the model would be 8.9k parameters (taking the less conservative 1:10 ratio of parameters to corpus tokens). However, given the specificity of the model, I contradict this ratio.

Given the risk of these models to heavily overfit, optimizing the ratio (even if there are more parameters than there are tokens) are critical. Therefore, I required another corpus of data that I did not have access to.

As a result, I turned to the strategy of using *Off Policy, Sequence Level Knowledge Distillation* from a larger model (Llama 3.2 1B). First, I finetuned the model on the corpus of meditations using [Unsloth AI's notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb).
Then, I used it to generate approximately 122k tokens of synthetic data over 100 iterations asking it common stoic prompts. I used Google Colab's inbuilt Tesla T4 GPU to run this data generation, which took about 1.5 hours.
Note: I did not run it from the project directory due to my lack of GPU configuration; however, the adapted notebook has been included.

One critical limitation in this approach is the inefficacy of my prompt library: I did not explicitly instruct the model to focus on _Meditations_ by Marcus Aurelius, enabling the model to hallucinate and pull from various sources of data.
Future iterations of AureliusGPT-Torch will account for this problem thoroughly to avoid brittle LoRA memory being further defeated, or move to another fine tuning/RL based technique. The core point of this corpus of data was to source philosophical, 
Stoic or Stoic adjacent data to ensure better generation quality.

My preprocessing logic between AureliusGPT-Torch and AureliusGPT is the same; I rely on Greek transliterations, Unicode normalization, regex patterns, and special "<BEGIN>", "<END>", and "<PAD>" (at training time) tokens.
I take a similar approach for my preprocessing of _Meditations_ corpus to feed into LoRA; to avoid confusion between Llama's internal tokens and mine, I avoid adding them in, instead semantically replacing them after the corpus has been generated.

I added the Meditations data twice to the final corpus to weight its significance, especially given its lower token count and the lower quality of synthetic data. My training split was 80% of the pure Meditations data and all the synthetic data; my validation split was 20% of the pure Meditations data.

## Architecture

### Overview
I use PyTorch's weight initialization (as opposed to a Gaussian process W ~ 𝒩(0, 0.02) for my manual AureliusGPT). I rely on the Transformer architecture from Attention is All You Need (Vaswani et al.) in my implementation.
Given the small scale of this project, all training (except for the Llama LoRA for synthetic data) was conducted on a local CPU, and was sequential (not concurrent, hence my lack of ThreadPool or ProcessPoolExecutor).
I rely on a PostNorm, 6 Transformer block architecture for AureliusGPT-Torch of the format

GPT -> {TransformerBlock -> TransformerBlock -> TransformerBlock...}x6 -> logits
TransformerBlock -> {AttentionBlock -> Add/Norm -> FFN -> Add/Norm}
AttentionBlock -> {AttentionHead + Concatenation}

### Embeddings
I use sinusoidal positional embeddings; given my low parameter budget, I thought it was economical to avoid learning them. My core embedding matrix E is learned.

### Attention
I use multihead attention in this project, as well as a causal attention mask.

### LayerNorm
I use PyTorch's inbuilt LayerNorm rather than my own implementation.


## Training

### Overview

As mentioned earlier, I incorporate a train/val split of 80:20. I also compute the training and validation loss, as well as the gradient normalization to track overfitting. 
num_epochs=50 is too high a number; you can using matplotlib graph the relevant losses and identify signs of overfitting during experiments. There is a clear route to On Policy Distillation
and feedback loops/PPO, RLHF, or RLAIF; there is a method in the original Unsloth AI notebook to save the LoRA weights of the _Meditations_ tuned teacher model to HuggingFace, which can be leveraged as a teacher 
model in the future. 

### Inference 

While I rely on Off Policy, Sequence Level Knowledge Distillation with LLAMA as outlined in my Data section, there is a clear route to implementing Best of N Sampling through concurrent model generations and rankings.
This can again rely on the finetuned LLAMA model or any comparable instruct model on its level.

This model, once fully completed, will be put on HuggingFace and hosted on my website (tarush.ai/aureliusgpt-torch). 
