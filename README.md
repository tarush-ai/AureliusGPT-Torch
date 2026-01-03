<h1 align="center">
  AureliusGPT-Torch
</h1>

<div align="center">

  <img src="https://raw.githubusercontent.com/tarush-ai/PersonalWebsite/refs/heads/main/aurelius-logo.png" alt="Aurelius Pixel Art" width="600" />
  
  <br />
  <br />

  [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000&style=for-the-badge)](https://huggingface.co/Tarush-AI/AureliusGPT)
  [![AureliusGPT UI](https://img.shields.io/badge/AureliusGPT_UI-111827?style=for-the-badge)](https://www.tarush.ai/aureliusgpt)

</div>

## Overview

AureliusGPT-Torch is an 845k, PyTorch and SentencePiece boosted SLM pretrained on _Meditations_ by Marcus Aurelius and other adjacent philosophical works. It is a larger size, more comprehensive reimplementation of 
[AureliusGPT](https://github.com/tarush-ai/AureliusGPT), a smaller model trained on the same core corpus, but using a handwritten/NumPy first (zero PyTorch or prewritten tokenizer) approach.

Rather than reimplementing a custom BPE algorithm and tokenizer backpropagation from scratch like its sibling repo, AureliusGPT-Torch trains SentencePiece on its corpus (detailed in Data) and relies on PyTorch for autograd.

The [HuggingFace](https://huggingface.co/Tarush-AI/AureliusGPT) for this model (including both the base model weights at epoch 10 and the tokenizer/vocabulary) are contained here. 

This work was not possible without [Project Gutemberg's opensource _Meditations_](https://www.gutenberg.org/cache/epub/2680/pg2680-images.html). A full license is listed at the end of this README.

## Usage

To get started, run:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you wish to have a "Stoic Validator," please create a .env file of the format:
```
OPENAI_API_KEY = sk_proj_YOUR_OPENAI_API_KEY_HERE
```

Alternatively, you can run from CLI and make the necessary changes:
```
export OPENAI_API_KEY=sk_proj_YOUR_OPENAI_API_KEY_HERE
```

A future version of this model will rely on the finetuned Llama 3.2 1B parameter model as the validator. 

To finetune the teacher model for distillation / synthetic data generation, open the Colab Notebook (model/datageneration/llama321b_meditations_lora_unsloth.ipynb) and run it on the provided Tesla T4 GPU. If the data has not been generated, training will rely on the core _Meditations_ corpus. Future versions of this repository will have the synthetic data generation pipeline / Unsloth package library wrapped into the core repository with GPU plugin/handling.

To run training from the source directory, run:
```
python -m model.train
```

To run the model in CLI, run:
```
python -m model.run
```

If you want to experiment with the preprocessing or tokenization logic in the corpus, .test() functions have been wrapped into both. You can try both:

```
python -m model.vocabulary.tokenizer test
python -m model.vocabulary.preprocess test 
```

Or, if you have your own unique testfile (this may yield errors due to the hyperspecific nature of these files:
```
python -m model.vocabulary.tokenizer test YOUR_RELATIVE_FILEPATH_HERE
python -m model.vocabulary.preprocess test YOUR_RELATIVE_FILEPATH_HERE
```
## Contribution

This project uses an MIT license. You are more than welcome to submit a pull request and contribute on this repository. My future plans for this repository are listed under the "Future Work on AureliusGPT" section, but as long as the pull request is conducive, I will accept it. You can also reach out to me [here](mailto:tarushgs@gmail.com) to join as a contributor to the repository.

## Data
The original corpus of _Meditations_ by Marcus Aurelius is 89k tokens approximately when tokenized by a SentencePiece BPE tokenizer trained on a vocabulary length of 2,000. Using Kaplan et al. Chinchilla scaling laws,
the expected parameter size of the model would be 8.9k parameters (taking the less conservative 1:10 ratio of parameters to corpus tokens). However, given the smaller size of the model and its lack of focus on general intelligence (instead, focused on generating Stoic-adjacent, Aurelius flavored text), this ratio does not apply.

Given the risk of these models to heavily overfit, optimizing the ratio (even if there are more parameters than there are tokens) is critical. Therefore, I required another corpus of data that I did not have access to.

As a result, I turned to the strategy of using *Off Policy, Sequence Level Knowledge Distillation* from a larger model (Llama 3.2 1B). First, I finetuned the model on the corpus of meditations using [Unsloth AI's notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb).
Then, I used it to generate approximately 122k tokens of synthetic data over 100 iterations asking it common stoic prompts. I used Google Colab's inbuilt Tesla T4 GPU to run this data generation, which took about 1.5 hours. I do not match logits or trajectories, only generated text; therefore, this approach also pulls elements from instruct-style distillation or offline student-teacher SFT methods. 
Note: I did not run it from the project directory due to my lack of GPU configuration; however, the adapted notebook has been included.

One critical limitation in this approach is the inefficacy of my prompt library: I did not explicitly instruct the model to focus on _Meditations_ by Marcus Aurelius, enabling the model to hallucinate and pull from various sources of data.
Future iterations of AureliusGPT-Torch will account for this problem thoroughly to avoid brittle LoRA memory being further defeated, or move to another fine tuning/RL based technique. The core point of this corpus of data was to source philosophical, 
Stoic or Stoic adjacent data to ensure better generation quality.

My preprocessing logic between AureliusGPT-Torch and AureliusGPT is the same; I rely on Greek transliterations, Unicode normalization, regex patterns, and special "<BEGIN>", "<END>", and "<PAD>" (at training time) tokens.
I take a similar approach for my preprocessing of _Meditations_ corpus to feed into LoRA; to avoid confusion between Llama 3.2 1B's internal tokens and mine, I avoid adding them in, instead semantically replacing them after the corpus has been generated.

I added the Meditations data twice to the final corpus to weight its significance, especially given its lower token count and the lower quality of synthetic data. My training split was 80% of the pure Meditations data and all the synthetic data; my validation split was 20% of the pure Meditations data.

## Architecture

### Overview
I use PyTorch's weight initialization (as opposed to a Gaussian process W ~ 𝒩(0, 0.02) for my manual AureliusGPT). I rely on the Transformer architecture from Attention is All You Need (Vaswani et al.) in my implementation.
Given the small scale of this project, all training (except for the Llama 3.2 1B LoRA for synthetic data) was conducted on a local CPU, and was sequential (not concurrent, hence my lack of ThreadPool or ProcessPoolExecutor).
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

While I rely on Off Policy, Sequence Level Knowledge Distillation with Llama 3.2 1B as outlined in my Data section, there is a clear route to implementing Best of N Sampling through concurrent model generations and rankings.
This can again rely on the finetuned Llama 3.2 1B model or any comparable instruct model on its level.

This model, once fully completed, will be put on HuggingFace and hosted on my website (tarush.ai/aureliusgpt-torch). 

## Future Work on AureliusGPT

### Synthetic Data Generation / Teacher Model

Currently, my LoRA'd Llama 3.2 1B model is run in an ipynb on a Tesla T4 GPU. A future version will integrate the LoRA and synthetic data generation, and relevant GPU plumbing, into the scope of this project. 

Additionally, the universal "teacher/Stoic justifier" model will be the adapted Llama 3.2 1B model, deviating from the OpenAI Chat Completions API GPT-4o approach. 

### Model Visibility on HuggingFace

In a future version, the fitting and accuracy optimized (see Overfitting Tracking / Adaptable Training) files of Llama 3.2 1B's LoRA weights and AureliusGPT-Torch will be loaded onto HuggingFace for fast import and inference.

### Top N Sampling

After the teacher model is converted to Llama 3.2 1B, I will implement config editable concurrent AureliusGPT generation and top N sampling to ensure the highest quality result.

### Overfitting Tracking / Adaptable Training

Currently, training highlights training and validation loss, as well as gradient normalization based on the train/test split to identify overfitting. In a future version, this will be tracked in an easy-to-interpret, modular plot for the user for ease of training.

Beyond weight tuning, config.py will be helpfully automatically adjusted in a future version, changing the learning rate, number of epochs, batch size, and other aspects of training. Therefore, after running n training cycles on its own, the model will iteratively improve its training performance so minimal training optimization is required. 

### Model Upscale

A future project will use the LoRA'd Llama 3.2 1B model to generate signifcantly more parameters of stoic adjacent text, as well as utilizing the works of Zeno, Epictetus, and other famous Stoic thinkers, to build either a Transformer or an MoE model ("Epictetus": _____, "Zeno": _______, "Aurelius": _______) called "StoicDiscourseLM". This will incorporate many elements (including preprocessing functionality) of AureliusGPT but will also be a unique project.


## Sample Outputs

### Overview

Some malformed characters and spacing artifacts are due to SentencePiece segmentation and Unicode normalization interactions at small scale, not post-processing errors. With an increased vocab size and optimized Unicode, these errors would not persist.

The config file used to generate these results is on GitHub, but here are the relevant stats:
```
vocab_size = 4000

num_blocks = 6
d_model = 64
h = 4
d_head = 16
d_ff = 64

#training time
batch_size = 512
num_epochs = 50
max_seq_length = 512
lr = 1e-3

#runtime
max_tokens = 200
temperature = 0.8
justification_model = "gpt-4"
argmax = False     
```


### Sample Output 1: "The world "

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

---

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

---

### Sample Output 3: "Pain is "

AureliusGPT

A model trained on Meditations by Marcus Aurelius.

User: Pain is 

AURELIUS: a common end. Whatsoever is kindly now can be, when the common proper use of all external accidents, as it doth as it be. They cannot be thy pray, doth his own proper and when are now properly be that it were that which above, even the Pl atonic, and upon this fort ified or if a man whosoever of the fig - father against reason with thee, and that common part of conv ulsion. But in matter of another; whether as that rest contented both resumed and with the things that his own operation.  ⁇ eep thyself, and all doth in others maintain.  ⁇ emember my youth s of the mind as in those that is an ext remity soever is a matter of other peculiar property, or fe ite, but for those things, nor the present themselves, and which is, hast taken out of it be the further ance of a work. That which at all things that which if it were the worse than we see with a perpetual change, and yet even with a w its of all things present ' s commonwealth, that which the satisf action of a man make rec an, it. See this is but as so much as in passion and the city, they should good things that thou such a happy man should ) so that thy refuge, is but best, nor, though but that which is to the Gods, is incidental unto that which is fitting; it be not to pass together, that that which is no more than this that is chiefest, and this, fear that is also is benef man as hitherto thou art find s neighbour should not in this doth himself, or some common good or friend ly, and what there is a man ' s nature that which we may be not in all other course of all this end is, or liberty or ators are the universe did he that both of man is an error, what is mere change? or evil. If therefore is the man likewise. Now all, and that is the series and un sp and the case and that which is either lot and make himself from the mind is now, that which is bound to be no hurt, if my possession of the several roll mutation and that which is not in his kind of all his neighbour is now be a man ' s uch is sociably and necessary? or a man, or a series of those that, that unto man may be a man either a man before after a man whose end; but a man, by man, earth; and the common to whom thy thoughts with this place to live long, and conceit and objects, honour, and re ward on ge song to depart avoid or dispersion, therein any man ste ady, thou didst not lust after one that neither with anything, or st y.  ⁇ es; and that which is proper, is sure that which is allotted unto man whose arm ho pes, I think in regard of all anger: whether all this nature, which is most justly be no man ' s turn. To be a man should rest and separation to be in a common earth, and all things, and duration and all things themselves it were not in some man to embrace whatsoever doth send; and that part and from it were transl ated; therefore, and all things that weight y him, and it, temperate, for their wills that are reasonable faculty, and another: from the examples of such a very little pro cure that which is unto her,) are in thy thoughts that is said, before. By so always be no evil; good and una ffect not grieve more fre quent public like one that the reason, is done, which is proper and what things, or pain ful both be performed to live according to represent pt ations of another is, and the use to do nothing against their pleasure, and opinions, but withdraw. Then long.  ⁇ ep id either the world. It is, either reasonable administration s work of an instant ), is end. What then at all hurtful unto the C ord ant; and not to want of all things: and this and how either form and commonwealth, is, or proper to all substances. But if he is it suffice from all kind and so are of those things in them sw pp ly my particular, and all resumed to man may be a very ridiculous as it is sociably and coherence of a man, why should endeavour anything else, and a certain end.  

 
Stoic Justification Agent: 
This output gets a score of 40/100. 

The model seems to have taken some elements from "Meditations", like the discussions of commonality, the state of being, mortality, and reason, which align fairly well with Stoic philosophies. For instance, Marcus Aurelius often contemplates the nature of humans, life, and existence in the universe, as well as postulates on proper rational behavior. 

However, these themes are drastically overshadowed by the model's incoherency. Semantically, it fails slightly because the output fails to follow a single coherent topic. Instead, there are sudden jumps from one idea to another resulting in lack of contextual associations between sentences. For example, the phrases like "the mind is now", "this end is", or "thy refuge, is but best, nor, though but that which is to the gods, is incidental unto" are hard or impossible to comprehend logically.

From a linguistic perspective, the model fails significantly. There's repetitive usage of certain phrases such as "that which is" and "man", which greatly hampers the text's readability. Moreover, the response is littered with disjointed phrases, narrative inconsistencies, grammatical errors, and meaningless symbols, which makes the text challenging to read and comprehend.

In the context of stoic teachings, while the text vaguely alludes to themes of mortality, human nature, and life's ephemity that Marcus Aurelius often touched upon, it fails to do so in a meaningful or insightful manner. The output hardly embodies stoicism's emphasis on logic, self-discipline, and practical wisdom.

In conclusion, while the model has some alignment with the overall themes of "Meditations", the output lacks coherence, linguistic accuracy, and effective conveyance of Stoic philosophy. Significant work is needed to improve on these areas.

---

# Project Gutemberg License

START: FULL LICENSE

THE FULL PROJECT GUTENBERG LICENSE

PLEASE READ THIS BEFORE YOU DISTRIBUTE OR USE THIS WORK

To protect the Project Gutenberg™ mission of promoting the free
distribution of electronic works, by using or distributing this work
(or any other work associated in any way with the phrase “Project
Gutenberg”), you agree to comply with all the terms of the Full
Project Gutenberg™ License available with this file or online at
www.gutenberg.org/license.

Section 1. General Terms of Use and Redistributing Project Gutenberg™
electronic works

1.A. By reading or using any part of this Project Gutenberg™
electronic work, you indicate that you have read, understand, agree to
and accept all the terms of this license and intellectual property
(trademark/copyright) agreement. If you do not agree to abide by all
the terms of this agreement, you must cease using and return or
destroy all copies of Project Gutenberg™ electronic works in your
possession. If you paid a fee for obtaining a copy of or access to a
Project Gutenberg™ electronic work and you do not agree to be bound
by the terms of this agreement, you may obtain a refund from the person
or entity to whom you paid the fee as set forth in paragraph 1.E.8.

1.B. “Project Gutenberg” is a registered trademark. It may only be
used on or associated in any way with an electronic work by people who
agree to be bound by the terms of this agreement. There are a few
things that you can do with most Project Gutenberg™ electronic works
even without complying with the full terms of this agreement. See
paragraph 1.C below. There are a lot of things you can do with Project
Gutenberg™ electronic works if you follow the terms of this
agreement and help preserve free future access to Project Gutenberg™
electronic works. See paragraph 1.E below.

1.C. The Project Gutenberg Literary Archive Foundation (“the
Foundation” or PGLAF), owns a compilation copyright in the collection
of Project Gutenberg™ electronic works. Nearly all the individual
works in the collection are in the public domain in the United
States. If an individual work is unprotected by copyright law in the
United States and you are located in the United States, we do not
claim a right to prevent you from copying, distributing, performing,
displaying or creating derivative works based on the work as long as
all references to Project Gutenberg are removed. Of course, we hope
that you will support the Project Gutenberg™ mission of promoting
free access to electronic works by freely sharing Project Gutenberg™
works in compliance with the terms of this agreement for keeping the
Project Gutenberg™ name associated with the work. You can easily
comply with the terms of this agreement by keeping this work in the
same format with its attached full Project Gutenberg™ License when
you share it without charge with others.

1.D. The copyright laws of the place where you are located also govern
what you can do with this work. Copyright laws in most countries are
in a constant state of change. If you are outside the United States,
check the laws of your country in addition to the terms of this
agreement before downloading, copying, displaying, performing,
distributing or creating derivative works based on this work or any
other Project Gutenberg™ work. The Foundation makes no
representations concerning the copyright status of any work in any
country other than the United States.

1.E. Unless you have removed all references to Project Gutenberg:

1.E.1. The following sentence, with active links to, or other
immediate access to, the full Project Gutenberg™ License must appear
prominently whenever any copy of a Project Gutenberg™ work (any work
on which the phrase “Project Gutenberg” appears, or with which the
phrase “Project Gutenberg” is associated) is accessed, displayed,
performed, viewed, copied or distributed:

    This eBook is for the use of anyone anywhere in the United States and most
    other parts of the world at no cost and with almost no restrictions
    whatsoever. You may copy it, give it away or re-use it under the terms
    of the Project Gutenberg License included with this eBook or online
    at www.gutenberg.org. If you
    are not located in the United States, you will have to check the laws
    of the country where you are located before using this eBook.
  
1.E.2. If an individual Project Gutenberg™ electronic work is
derived from texts not protected by U.S. copyright law (does not
contain a notice indicating that it is posted with permission of the
copyright holder), the work can be copied and distributed to anyone in
the United States without paying any fees or charges. If you are
redistributing or providing access to a work with the phrase “Project
Gutenberg” associated with or appearing on the work, you must comply
either with the requirements of paragraphs 1.E.1 through 1.E.7 or
obtain permission for the use of the work and the Project Gutenberg™
trademark as set forth in paragraphs 1.E.8 or 1.E.9.

1.E.3. If an individual Project Gutenberg™ electronic work is posted
with the permission of the copyright holder, your use and distribution
must comply with both paragraphs 1.E.1 through 1.E.7 and any
additional terms imposed by the copyright holder. Additional terms
will be linked to the Project Gutenberg™ License for all works
posted with the permission of the copyright holder found at the
beginning of this work.

1.E.4. Do not unlink or detach or remove the full Project Gutenberg™
License terms from this work, or any files containing a part of this
work or any other work associated with Project Gutenberg™.

1.E.5. Do not copy, display, perform, distribute or redistribute this
electronic work, or any part of this electronic work, without
prominently displaying the sentence set forth in paragraph 1.E.1 with
active links or immediate access to the full terms of the Project
Gutenberg™ License.

1.E.6. You may convert to and distribute this work in any binary,
compressed, marked up, nonproprietary or proprietary form, including
any word processing or hypertext form. However, if you provide access
to or distribute copies of a Project Gutenberg™ work in a format
other than “Plain Vanilla ASCII” or other format used in the official
version posted on the official Project Gutenberg™ website
(www.gutenberg.org), you must, at no additional cost, fee or expense
to the user, provide a copy, a means of exporting a copy, or a means
of obtaining a copy upon request, of the work in its original “Plain
Vanilla ASCII” or other form. Any alternate format must include the
full Project Gutenberg™ License as specified in paragraph 1.E.1.

1.E.7. Do not charge a fee for access to, viewing, displaying,
performing, copying or distributing any Project Gutenberg™ works
unless you comply with paragraph 1.E.8 or 1.E.9.

1.E.8. You may charge a reasonable fee for copies of or providing
access to or distributing Project Gutenberg™ electronic works
provided that:

    • You pay a royalty fee of 20% of the gross profits you derive from
        the use of Project Gutenberg™ works calculated using the method
        you already use to calculate your applicable taxes. The fee is owed
        to the owner of the Project Gutenberg™ trademark, but he has
        agreed to donate royalties under this paragraph to the Project
        Gutenberg Literary Archive Foundation. Royalty payments must be paid
        within 60 days following each date on which you prepare (or are
        legally required to prepare) your periodic tax returns. Royalty
        payments should be clearly marked as such and sent to the Project
        Gutenberg Literary Archive Foundation at the address specified in
        Section 4, “Information about donations to the Project Gutenberg
        Literary Archive Foundation.”
    
    • You provide a full refund of any money paid by a user who notifies
        you in writing (or by e-mail) within 30 days of receipt that s/he
        does not agree to the terms of the full Project Gutenberg™
        License. You must require such a user to return or destroy all
        copies of the works possessed in a physical medium and discontinue
        all use of and all access to other copies of Project Gutenberg™
        works.
    
    • You provide, in accordance with paragraph 1.F.3, a full refund of
        any money paid for a work or a replacement copy, if a defect in the
        electronic work is discovered and reported to you within 90 days of
        receipt of the work.
    
    • You comply with all other terms of this agreement for free
        distribution of Project Gutenberg™ works.
    

1.E.9. If you wish to charge a fee or distribute a Project
Gutenberg™ electronic work or group of works on different terms than
are set forth in this agreement, you must obtain permission in writing
from the Project Gutenberg Literary Archive Foundation, the manager of
the Project Gutenberg™ trademark. Contact the Foundation as set
forth in Section 3 below.

1.F.

1.F.1. Project Gutenberg volunteers and employees expend considerable
effort to identify, do copyright research on, transcribe and proofread
works not protected by U.S. copyright law in creating the Project
Gutenberg™ collection. Despite these efforts, Project Gutenberg™
electronic works, and the medium on which they may be stored, may
contain “Defects,” such as, but not limited to, incomplete, inaccurate
or corrupt data, transcription errors, a copyright or other
intellectual property infringement, a defective or damaged disk or
other medium, a computer virus, or computer codes that damage or
cannot be read by your equipment.

1.F.2. LIMITED WARRANTY, DISCLAIMER OF DAMAGES - Except for the “Right
of Replacement or Refund” described in paragraph 1.F.3, the Project
Gutenberg Literary Archive Foundation, the owner of the Project
Gutenberg™ trademark, and any other party distributing a Project
Gutenberg™ electronic work under this agreement, disclaim all
liability to you for damages, costs and expenses, including legal
fees. YOU AGREE THAT YOU HAVE NO REMEDIES FOR NEGLIGENCE, STRICT
LIABILITY, BREACH OF WARRANTY OR BREACH OF CONTRACT EXCEPT THOSE
PROVIDED IN PARAGRAPH 1.F.3. YOU AGREE THAT THE FOUNDATION, THE
TRADEMARK OWNER, AND ANY DISTRIBUTOR UNDER THIS AGREEMENT WILL NOT BE
LIABLE TO YOU FOR ACTUAL, DIRECT, INDIRECT, CONSEQUENTIAL, PUNITIVE OR
INCIDENTAL DAMAGES EVEN IF YOU GIVE NOTICE OF THE POSSIBILITY OF SUCH
DAMAGE.

1.F.3. LIMITED RIGHT OF REPLACEMENT OR REFUND - If you discover a
defect in this electronic work within 90 days of receiving it, you can
receive a refund of the money (if any) you paid for it by sending a
written explanation to the person you received the work from. If you
received the work on a physical medium, you must return the medium
with your written explanation. The person or entity that provided you
with the defective work may elect to provide a replacement copy in
lieu of a refund. If you received the work electronically, the person
or entity providing it to you may choose to give you a second
opportunity to receive the work electronically in lieu of a refund. If
the second copy is also defective, you may demand a refund in writing
without further opportunities to fix the problem.

1.F.4. Except for the limited right of replacement or refund set forth
in paragraph 1.F.3, this work is provided to you ‘AS-IS’, WITH NO
OTHER WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO WARRANTIES OF MERCHANTABILITY OR FITNESS FOR ANY PURPOSE.

1.F.5. Some states do not allow disclaimers of certain implied
warranties or the exclusion or limitation of certain types of
damages. If any disclaimer or limitation set forth in this agreement
violates the law of the state applicable to this agreement, the
agreement shall be interpreted to make the maximum disclaimer or
limitation permitted by the applicable state law. The invalidity or
unenforceability of any provision of this agreement shall not void the
remaining provisions.

1.F.6. INDEMNITY - You agree to indemnify and hold the Foundation, the
trademark owner, any agent or employee of the Foundation, anyone
providing copies of Project Gutenberg™ electronic works in
accordance with this agreement, and any volunteers associated with the
production, promotion and distribution of Project Gutenberg™
electronic works, harmless from all liability, costs and expenses,
including legal fees, that arise directly or indirectly from any of
the following which you do or cause to occur: (a) distribution of this
or any Project Gutenberg™ work, (b) alteration, modification, or
additions or deletions to any Project Gutenberg™ work, and (c) any
Defect you cause.

Section 2. Information about the Mission of Project Gutenberg™

Project Gutenberg™ is synonymous with the free distribution of
electronic works in formats readable by the widest variety of
computers including obsolete, old, middle-aged and new computers. It
exists because of the efforts of hundreds of volunteers and donations
from people in all walks of life.

Volunteers and financial support to provide volunteers with the
assistance they need are critical to reaching Project Gutenberg™’s
goals and ensuring that the Project Gutenberg™ collection will
remain freely available for generations to come. In 2001, the Project
Gutenberg Literary Archive Foundation was created to provide a secure
and permanent future for Project Gutenberg™ and future
generations. To learn more about the Project Gutenberg Literary
Archive Foundation and how your efforts and donations can help, see
Sections 3 and 4 and the Foundation information page at www.gutenberg.org.

Section 3. Information about the Project Gutenberg Literary Archive Foundation

The Project Gutenberg Literary Archive Foundation is a non-profit
501(c)(3) educational corporation organized under the laws of the
state of Mississippi and granted tax exempt status by the Internal
Revenue Service. The Foundation’s EIN or federal tax identification
number is 64-6221541. Contributions to the Project Gutenberg Literary
Archive Foundation are tax deductible to the full extent permitted by
U.S. federal laws and your state’s laws.

The Foundation’s business office is located at 809 North 1500 West,
Salt Lake City, UT 84116, (801) 596-1887. Email contact links and up
to date contact information can be found at the Foundation’s website
and official page at www.gutenberg.org/contact

Section 4. Information about Donations to the Project Gutenberg
Literary Archive Foundation

Project Gutenberg™ depends upon and cannot survive without widespread
public support and donations to carry out its mission of
increasing the number of public domain and licensed works that can be
freely distributed in machine-readable form accessible by the widest
array of equipment including outdated equipment. Many small donations
($1 to $5,000) are particularly important to maintaining tax exempt
status with the IRS.

The Foundation is committed to complying with the laws regulating
charities and charitable donations in all 50 states of the United
States. Compliance requirements are not uniform and it takes a
considerable effort, much paperwork and many fees to meet and keep up
with these requirements. We do not solicit donations in locations
where we have not received written confirmation of compliance. To SEND
DONATIONS or determine the status of compliance for any particular state
visit www.gutenberg.org/donate.

While we cannot and do not solicit contributions from states where we
have not met the solicitation requirements, we know of no prohibition
against accepting unsolicited donations from donors in such states who
approach us with offers to donate.

International donations are gratefully accepted, but we cannot make
any statements concerning tax treatment of donations received from
outside the United States. U.S. laws alone swamp our small staff.

Please check the Project Gutenberg web pages for current donation
methods and addresses. Donations are accepted in a number of other
ways including checks, online payments and credit card donations. To
donate, please visit: www.gutenberg.org/donate.

Section 5. General Information About Project Gutenberg™ electronic works

Professor Michael S. Hart was the originator of the Project
Gutenberg™ concept of a library of electronic works that could be
freely shared with anyone. For forty years, he produced and
distributed Project Gutenberg™ eBooks with only a loose network of
volunteer support.

Project Gutenberg™ eBooks are often created from several printed
editions, all of which are confirmed as not protected by copyright in
the U.S. unless a copyright notice is included. Thus, we do not
necessarily keep eBooks in compliance with any particular paper
edition.

Most people start at our website which has the main PG search
facility: www.gutenberg.org.

This website includes information about Project Gutenberg™,
including how to make donations to the Project Gutenberg Literary
Archive Foundation, how to help produce our new eBooks, and how to
subscribe to our email newsletter to hear about new eBooks.
