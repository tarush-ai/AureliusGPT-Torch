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

## Sample Outputs

### Sample Output 1: "The World"

AureliusGPT

A model trained on Meditations by Marcus Aurelius.

User: The world

AURELIUS: for us, when therefore even in all shall never be eth, to have no thought to believe not say unto me. If it will this kind of them to strive and by right, is, what respect and that, or wise as a present station, yea it be with such a man, and opportunity to thyself unto himself from all these ever to be bound to be hindered; and to his own proper natural dreams I cannot be his mind. Thou must needs any man. Be get ative faculty, and see whatsoever it is of a kind of the true and reason, and nature of the general condition of the atre and affections to live according to any man ' s rational substance, a man doth subject to live that seemed nor bad and of comm ent orderly man ' s apparent that is nothing more moderate cond escend ing to the mind and so towards him that was made for a man might comp are leisure to teach with ance of an honest! This ph ed ing to live, and knowledge, only.  

 
Stoic Justification Agent: 
Score: 60/100

Accuracy Report:
The model's generation has semantic, linguistic and Stoic-based connections with Marcus Aurelius' Meditations, but also presents a number of linguistic inaccuracies and inconsistencies. The text maintains a solemn and introspective tone akin to the Stoic philosophy depicted in Meditations, with phrases such as "rational substance," "natural dreams," and "general condition," all of which could plausibly reflect Aurelius’ contemplations on life, existence, and virtue.

Some sentences or phrases, like "Be get ative faculty, and see whatsoever it is of a kind of the true and reason", seem to bear traces of Stoic themes, but their meaning is encumbered by grammatical errors and awkward wording, lowering the overall score.

The phrase "opportunity to thyself unto himself" seems to echo Stoic teachings about self-reliance and introspection, while "man doth subject to live that seemed nor bad and of comm ent orderly man" and "a man might comp are leisure to teach with ance of an honest!" are hard to parse linguistically and thematically, bringing down the score further.

Overall, while the output bears semblances to the Stoic philosophy and style of Marcus Aurelius, the linguistic inaccuracies and incoherencies limit the model's score. Improvements could be made through refining the model's grasp of grammatical structures and semantic context.

### Sample Output 2: "Death "

User: Death

AURELIUS: if one that which ever not to some common good things con ven eth ents unto him, and the whole, but as concer t and the generation of what is inf ition of death dwelt with a blow as it with w ounded which is fitting; but a man would it be said to the vulgar apprehend the several of the feet, and solid th? set to the earth of pleasures? an act ate man ' s upon it, every man as a fool too? when matters, and whatsoever else whereof the whole series of blood of imp ure from all things are very truth, if thou wouldst thou dost think un as to the ext ortion again that perfect simplicity. If thou live with a kos mos or to say, and familiarly ue; and to the common good, and what things in very little while, and as that action? and of time? Those things anim ated into the matter of another, what manner of moment. If thou hast thou art now present are that to nature. What is done. Cont rary: is so must all things in regard the end of the common substance, and who otherwise it, and righteousness, and give thank that can live into perform all agree with it, and according to lose a man, and absolutely and what is a reasonable, and the whole world? and is properly. To a man, that knows not seen, is proper unto thee? towards thy flesh; that is but a rational sociable, and rational, towards them. Why should be a flood and in thy sympathy with it may but draw thy thoughts and credit must be hindered from the common good things that are they doth resist ed ing. For all things. And yet they have done, upon every one point, who is the same kind.  

 
Stoic Justification Agent: 
Score: 45/100

Accuracy Report:

The model's response is a complex body of text with a lot of various elements involved. Parts of it seem to be relating to stoicism's thoughts on death, the interconnectedness of all things, living in the moment and accepting our natural progression towards death. Words that fit with Aurelius's meditations like "common good", "rational", "nature", "simplicity", and questions about life's actions and the nature of living beings are included, showing the model's capability to generate stoic-based content in line with the author.

However, the output does lose points for coherency and linguistic accuracy - it seems to be piecing together phrases and ideas without a clear flow or a strong connection to the initial prompt of "Death". The composition is difficult to comprehend fully, with several instances of phrases being cut and jumbled up, affecting the readability and comprehension. This creates a somewhat disjointed narrative that might not offer the user a satisfactory response to their prompt.

In semantic terms, the toy model presents various stoic elements like maintaining virtues, contemplating nature, and contemplating life and death, which were central themes in the meditations of Marcus Aurelius. 

The model also presents some reflections on human nature, societal matters, and philosophy, showing its ability to generate content from the training material. However, these reflections are scattered throughout and don't weave into a coherent message.

Overall, considering this is a toy model and its limitations, the generated output shows some application of the stoic principles as seen in Meditations by Marcus Aurelius, but lacks the polish and coherence of the original text.