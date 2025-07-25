## Model Details
This section provides fundamental information about the model, helping stakeholders understand its context and key characteristics.

### Person or organization developing model:
**Developed by:** BigScience ([website](https://bigscience.huggingface.co))

*All collaborators are either volunteers or have an agreement with their employer. (Further breakdown of participants forthcoming.)*

**Funded by:**
    
* The French government.

* Hugging Face ([website](https://huggingface.co)).

* Organizations of contributors.  *(Further breakdown of organizations forthcoming.)*

### Model date:
Version 1.3 / 6 July 2022

Release Date Estimate: Monday, 11.July.2022

Started 11th March, 2022 11:42am PST

Estimated end: 5th July, 2022

### Model version:
Version 1.0.0

Version 1.3

### Model type:
Transformer-based Language Model

Autoregressive Large Language Model (LLM)

Decoder-only architecture

176,247,271,424 parameters:

* 3,596,615,680 embedding parameters

* 70 layers, 112 attention heads

* Hidden layers are 14336-dimensional

Sequence length of 2048 tokens used (see [BLOOM tokenizer](https://huggingface.co/bigscience/tokenizer), [tokenizer description](#tokenization))

Checkpoints format: `transformers` (Megatron-DeepSpeed format available [here](https://huggingface.co/bigscience/bloom-optimizer-states))

Languages: Multiple; see [training data](#training-data)

Modified from Megatron-LM GPT2 (see [paper](https://arxiv.org/abs/1909.08053), [BLOOM Megatron code](https://github.com/bigscience-workshop/Megatron-DeepSpeed)):

Layer normalization applied to word embeddings layer (`StableEmbedding`; see [code](https://github.com/facebookresearch/bitsandbytes), [paper](https://arxiv.org/pdf/2110.02861.pdf))

ALiBI positional encodings (see [paper](https://arxiv.org/pdf/2108.12409.pdf)), with GeLU activation functions

### Training details:
**Objective Function:** Cross Entropy with mean reduction (see [API documentation](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss)).

Training Iteration  95000

Total seen tokens: **366B**

Number of epochs: 1

### Paper or other resource for more information:
Link to paper: [here](https://arxiv.org/abs/2211.05100)

Please see [the BLOOM training README](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr11-176B-ml#readme) for full details on replicating training.

Training logs: [Tensorboard link](https://huggingface.co/tensorboard/bigscience/tr11-176B-ml-logs/)

More details on the architecture/optimizer: https://github.com/bigscience-workshop/bigscience/tree/master/train/tr11-176B-ml

Details on the distributed setup used for the training: https://github.com/bigscience-workshop/bigscience/tree/master/train/tr11-176B-ml

Tensorboard updated during the training: https://huggingface.co/bigscience/tr11-176B-ml-logs/tensorboard#scalars&tagFilter=loss

Blog post detailing the design choices during the dataset creation: https://bigscience.huggingface.co/blog/building-a-tb-scale-multilingual-dataset-for-language-modeling

Blog post summarizing how the architecture, size, shape, and pre-training duration where selected: https://bigscience.huggingface.co/blog/what-language-model-to-train-if-you-have-two-million-gpu-hours

Blog post on the hardware/engineering side: https://bigscience.huggingface.co/blog/which-hardware-to-train-a-176b-parameters-model

Lessons learned: https://github.com/bigscience-workshop/bigscience/blob/master/train/lessons-learned.md

Details on the obstacles overcome during the preparation on the engineering side: https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md

Initial prompting experiments using interim checkpoints: https://huggingface.co/spaces/bigscience/bloom-book

Intermediate checkpoints: https://huggingface.co/bigscience/bloom-176-intermediate

Original checkpoints repo for Megatron-DeepSpeed format: https://huggingface.co/bigscience/bloom-optimizer-states

Many intermediate checkpoints are available at https://huggingface.co/bigscience/bloom-intermediate/

### Citation details:
Cite as: BigScience, _BigScience Language Open-science Open-access Multilingual (BLOOM) Language Model_. International, May 2021-May 2022

### License:
RAIL License v1.0 ([link](https://huggingface.co/spaces/bigscience/license) / [article and FAQ](https://bigscience.huggingface.co/blog/the-bigscience-rail-license))

See the [BLOOM License](https://huggingface.co/spaces/bigscience/license), Attachment A, for detailed usage restrictions.

### Contact:
Send Questions to: bigscience-contact@googlegroups.com

---

## Intended Use
This section outlines the intended applications of the model.

### Primary intended uses:
Text generation

Exploring characteristics of language generated by a language model

Examples: Cloze tests, counterfactuals, generations with reframings

Tasks that leverage language models include: Information Extraction, Question Answering, Summarization

Enable public research on large language models (LLMs).

LLMs are intended to be used for language generation or as a pretrained base model that can be further fine-tuned for specific tasks. Use cases below are not exhaustive.

This model is being created in order to enable public research on large language models (LLMs).

### Primary intended users:
General Public

Researchers

Students

Educators

Engineers/developers

Non-commercial entities

Community advocates, including human and civil rights groups

Users of derivatives created by Direct Users, such as those using software with an [intended use](#intended-use)

Users of [Derivatives of the Model, as described in the License](https://huggingface.co/spaces/bigscience/license)

### Out-of-scope uses:
Using the model in [high-stakes](#high-stakes) settings is out of scope for this model.  The model is not designed for [critical decisions](#critical-decisions) nor uses with any material consequences on an individual's livelihood or wellbeing. The model outputs content that appears factual but may not be correct.

Out-of-scope Uses Include:

* Usage in biomedical domains, political and legal domains, or finance domains

* Usage for evaluating or scoring individuals, such as for employment, education, or credit

* Applying the model for critical automatic decisions, generating factual content, creating reliable summaries, or generating predictions that must be correct

---

## How to Use
This section outlines how to use the model.

This model can be easily used and deployed using HuggingFace's ecosystem. This needs `transformers` and `accelerate` installed. The model can be downloaded as follows:

 <img src="https://s3.amazonaws.com/moonup/production/uploads/1657271608456-62441d1d9fdefb55a0b7d12c.png" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>

---

## Factors
This section addresses variables that may impact the model's performance.

### Relevant factors:
Language, such as English or Yoruba

Domain, such as newswire or stories

Demographic characteristics, such as gender or nationality

### Evaluation factors:
Language, such as English or Yoruba

Domain, such as newswire or stories

Demographic characteristics, such as gender or nationality

---

## Metrics
This section describes how the model's performance is evaluated.

### Model performance measures:
| Metric             | Why chosen                                                         |
|--------------------|--------------------------------------------------------------------|
| [Perplexity](#perplexity)         | Standard metric for quantifying model improvements during training |
| Cross Entropy [Loss](#loss) | Standard objective for language models.                            |

And multiple different metrics for specific tasks. _(More evaluation metrics forthcoming upon completion of evaluation protocol.)_

pass@1 ↑
pass@10 ↑
pass@100 ↑

### Decision thresholds:
Not available.

### Variation approaches:
Not available.

---

## Evaluation Data
This section provides details about the datasets used to evaluate the model.

### Datasets:
Not available.

### Motivation:
Not available.

### Preprocessing:
Not available.

---

## Training Data
This section provides details about the datasets used to train the model.

### Datasets:
Training data includes:

-   46 natural languages
    
-   13 programming languages

-   In 1.6TB of pre-processed text, converted into 350B unique tokens (see [the tokenizer section](#tokenization) for more.)

Details for each dataset are provided in individual [Data Cards](https://huggingface.co/spaces/bigscience/BigScienceCorpus), and the sizes of each of their contributions to the aggregated training data are presented in an [Interactive Corpus Map](https://huggingface.co/spaces/bigscience-catalogue-lm-data/corpus-map).

### Motivation:
Not available.

### Preprocessing:
**Tokenization:** The BLOOM tokenizer ([link](https://huggingface.co/bigscience/tokenizer)), a learned subword tokenizer trained using:
    
- A byte-level Byte Pair Encoding (BPE) algorithm 

- A simple pre-tokenization rule, no normalization

- A vocabulary size of 250,680

It was trained on a subset of a preliminary version of the corpus using alpha-weighting per language.

---

## Quantitative Analyses
This section presents disaggregated evaluation results.

### Unitary results:
**Zero-shot evaluations:**

<span style="color:red"><b>WARNING:</b> This section used to contain much more results, however they were not correct and we released without the approval of the evaluation working group. We are currently in the process of fixing the evaluations.</span>

See this repository for JSON files: https://github.com/bigscience-workshop/evaluation-results

| Task | Language | Metric | BLOOM-176B | OPT-175B* |
|:--------|:-----------------|:------------------------|-------------:|------------:|
| humaneval | python | pass@1 ↑ | 0.155 | 0.0 |
| humaneval | python | pass@10 ↑ | 0.328 | 0.0 |
| humaneval | python | pass@100 ↑ | 0.572 | 0.003 |


**Train-time Evaluation:**

Final checkpoint after 95K steps:

- Training Loss: 1.939

- Validation Loss: 2.061

- Perplexity: 7.045

For more see: https://huggingface.co/bigscience/tr11-176B-ml-logs

### Intersectional results:
Not available.

---

## Memory or Hardware Requirements
This section outlines the memory or hardware requirements for loading, deploying, and training the model.

### Loading Requirements:
Bf16 weights: 329GB

Full checkpoint with optimizer states: 2.3TB

### Deploying Requirements:
Not available.

### Training or Fine-tuning Requirements:
#### Hardware

* 384 A100 80GB GPUs (48 nodes)
    
* Additional 32 A100 80GB GPUs (4 nodes) in reserve

* 8 GPUs per node Using NVLink 4 inter-gpu connects, 4 OmniPath links

* CPU: AMD

* CPU memory: 512GB per node

* GPU memory: 640GB per node

* Inter-node connect: Omni-Path Architecture (OPA)

* NCCL-communications network: a fully dedicated subnet

* Disc IO network: shared network with other types of nodes

#### Software

* Megatron-DeepSpeed ([Github link](https://github.com/bigscience-workshop/Megatron-DeepSpeed))

* DeepSpeed ([Github link](https://github.com/microsoft/DeepSpeed))

* PyTorch (pytorch-1.11 w/ CUDA-11.5; see [Github link](https://github.com/pytorch/pytorch))

* apex ([Github link](https://github.com/NVIDIA/apex))

Compute infrastructure: Jean Zay Public Supercomputer, provided by the French government (see [announcement](https://www.enseignementsup-recherche.gouv.fr/fr/signature-du-marche-d-acquisition-de-l-un-des-supercalculateurs-les-plus-puissants-d-europe-46733)).

Server training location: Île-de-France, France

---

## Ethical Considerations
This section discusses the ethical considerations in model development, including challenges, risks, and solutions.

Sensitive data: Model may Contain [personal information](#personal-data-and-information)

Potential risks: Model may:

-   Overrepresent some viewpoints and underrepresent others

-   Contain stereotypes
  
-   Contain [personal information](#personal-data-and-information)

-   Generate:

    -   Hateful, abusive, or violent language

    -   Discriminatory or prejudicial language

    -   Content that may not be appropriate for all settings, including sexual content

-   Make errors, including producing incorrect information as if it were factual

-   Generate irrelevant or repetitive outputs

-   Induce users into attributing human traits to it, such as sentience or consciousness

Risk mitigation strategies: See the [BLOOM License](https://huggingface.co/spaces/bigscience/license), Attachment A, for detailed usage restrictions.

Misuse: Intentionally using the model for harm, violating [human rights](#human-rights), or other kinds of malicious activities, is a misuse of this model. This includes:

-   Spam generation

-   Disinformation and influence operations

-   Disparagement and defamation

-   Harassment and abuse
  
-   [Deception](#deception)

-   Unconsented impersonation and imitation

-   Unconsented surveillance 

-   Generating content without attribution to the model, as specified in the [RAIL License, Use Restrictions](https://huggingface.co/spaces/bigscience/license)

---

## Caveats and Recommendations
This section lists unresolved issues and provides guidance for users.

### Caveats:
<span style="color:red"><b>WARNING:</b> This section used to contain much more results, however they were not correct and we released without the approval of the evaluation working group. We are currently in the process of fixing the evaluations.</span>

Estimated carbon emissions:  *(Forthcoming.)*
    
Estimated electricity usage: *(Forthcoming.)*

### Recommendations:
-   Indirect users should be made aware when the content they're working with is created by the LLM.

-   Users should be aware of [Risks and Limitations](#risks-and-limitations), and include an appropriate age disclaimer or blocking interface as necessary.

-   Models trained or finetuned downstream of BLOOM LM should include an updated Model Card.

-   Users of the model should provide mechanisms for those affected to provide feedback, such as an email address for comments.

---

## Additional Information
<img src="https://cdn-uploads.huggingface.co/production/uploads/1657124309515-5f17f0a0925b9863e28ad517.png" alt="BigScience Logo" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>

BigScience Large Open-science Open-access Multilingual Language Model

Version 1.3 / 6 July 2022

Current Checkpoint: **Training Iteration  95000**

Link to paper: [here](https://arxiv.org/abs/2211.05100)

---

# Model Details

## Basics
*This section provides information about the model type, version, license, funders, release date, developers, and contact information.*
*It is useful for anyone who wants to reference the model.*

<details>
<summary>Click to expand</summary>

</details>


## Technical Specifications
*This section includes details about the model objective and architecture, and the compute infrastructure.*
*It is useful for people interested in model development.*

<details>
<summary>Click to expand</summary>

</details>

---

# Training
*This section provides information about the training data, the speed and size of training elements, and the environmental impact of training.*
*It is useful for people who want to learn more about the model inputs and training footprint.*

<details>
<summary>Click to expand</summary>

## Training Data
*This section provides a high-level overview of the training data. It is relevant for anyone who wants to know the basics of what the model is learning.*

</details>

## Speeds, Sizes, Times

Training throughput: About 150 TFLOP per GPU per second

Estimated cost of training: Equivalent of $2-5M in cloud computing (including preliminary experiments)

The training supercomputer, Jean Zay ([website](http://www.idris.fr/eng/jean-zay/jean-zay-presentation-eng.html)), uses mostly nuclear energy. The heat generated by it is reused for heating campus housing.

</details>

---

# Uses
*This section addresses questions around how the model is intended to be used, discusses the foreseeable users of the model (including those affected by the model), and describes uses that are considered out of scope or misuse of the model.*
*It is useful for anyone considering using the model or who is affected by the model.*

<details>
<summary>Click to expand</summary>

## Intended Use

### Direct Use

-   Text generation

-   Exploring characteristics of language generated by a language model

    -   Examples: Cloze tests, counterfactuals, generations with reframings

### Downstream Use

-   Tasks that leverage language models include: Information Extraction, Question Answering, Summarization

### Misuse and Out-of-scope Use
*This section addresses what users ought not do with the model.*

#### Out-of-scope Uses

#### Misuse

## Intended Users

### Direct Users

### Indirect Users

### Others Affected (Parties Prenantes)

-   People and groups referred to by the LLM

-   People and groups exposed to outputs of, or decisions based on, the LLM

-   People and groups whose original work is included in the LLM

</details>

---

# Risks and Limitations
*This section identifies foreseeable harms and misunderstandings.*
    
<details>
<summary>Click to expand</summary>

</details>

---

# Evaluation
*This section describes the evaluation protocols and provides the results.*


<details>
<summary>Click to expand</summary>

## Metrics
*This section describes the different ways performance is calculated and why.*

Includes:

## Factors
*This section lists some different aspects of BLOOM models. Its focus is on aspects that are likely to give rise to high variance in model behavior.*

##  Results
*Results are based on the [Factors](#factors) and [Metrics](#metrics).*

**Zero-shot evaluations:**

<span style="color:red"><b>WARNING:</b> This section used to contain much more results, however they were not correct and we released without the approval of the evaluation working group. We are currently in the process of fixing the evaluations.</span>

See this repository for JSON files: https://github.com/bigscience-workshop/evaluation-results

**Train-time Evaluation:**

</details>

---

# Recommendations

*This section provides information on warnings and potential mitigations.*

<details>
<summary>Click to expand</summary>

</details>

---

# Glossary and Calculations

*This section defines common terms and how metrics are calculated.*
<details>
<summary>Click to expand</summary>

-   <a name="loss">**Loss:**</a> A calculation of the difference between what the model has learned and what the data shows ("groundtruth"). The lower the loss, the better. The training process aims to minimize the loss.

-   <a name="perplexity">**Perplexity:**</a> This is based on what the model estimates the probability of new data is. The lower the perplexity, the better.  If the model is 100% correct at predicting the next token it will see, then the perplexity is 1. Mathematically this is calculated using entropy.

-   <a name="high-stakes">**High-stakes settings:**</a> Such as those identified as "high-risk AI systems" and "unacceptable risk AI systems" in the European Union's proposed [Artificial Intelligence (AI) Act](https://artificialintelligenceact.eu/annexes/).

-   <a name="critical-decisions">**Critical decisions:**</a> Such as those defined in [the United States' proposed Algorithmic Accountability Act](https://www.congress.gov/117/bills/s3572/BILLS-117s3572is.pdf).

-   <a name="human-rights">**Human rights:**</a> Includes those rights defined in the [Universal Declaration of Human Rights](https://www.un.org/sites/un2.un.org/files/2021/03/udhr.pdf).

-  <a name="personal-data-and-information">**Personal Data and Personal Information:**</a> Personal data and information is defined in multiple data protection regulations, such as "[personal data](https://gdpr-info.eu/issues/personal-data/)" in the [European Union's General Data Protection Regulation](https://gdpr-info.eu); and "personal information" in the Republic of South Africa's [Protection of Personal Information Act](https://www.gov.za/sites/default/files/gcis_document/201409/3706726-11act4of2013popi.pdf), The People's Republic of China's [Personal information protection law](http://en.npc.gov.cn.cdurl.cn/2021-12/29/c_694559.htm).

- <a name="sensitive-characteristics">**Sensitive characteristics:**</a> This includes specifically protected categories in human rights (see [UHDR, Article 2](https://www.un.org/sites/un2.un.org/files/2021/03/udhr.pdf)) and personal information regulation (see GDPR, [Article 9; Protection of Personal Information Act, Chapter 1](https://www.gov.za/sites/default/files/gcis_document/201409/3706726-11act4of2013popi.pdf))

- <a name="deception">**Deception:**</a> Doing something to intentionally mislead individuals to believe something that is false, such as by creating deadbots or chatbots on social media posing as real people, or generating text documents without making consumers aware that the text is machine generated.

</details>

---

# More Information
*This section provides links to writing on dataset creation, technical specifications, lessons learned, and initial results.*

<details>
<summary>Click to expand</summary>

## Intermediate checkpoints

For academic (or any) usage, we published the intermediate checkpoints, corresponding to the model state at each 5000 steps. Please follow [this link](https://huggingface.co/bigscience/bloom-176-intermediate) to get these checkpoints.

    
## Dataset Creation

Blog post detailing the design choices during the dataset creation: https://bigscience.huggingface.co/blog/building-a-tb-scale-multilingual-dataset-for-language-modeling

## Technical Specifications

Blog post summarizing how the architecture, size, shape, and pre-training duration where selected: https://bigscience.huggingface.co/blog/what-language-model-to-train-if-you-have-two-million-gpu-hours

More details on the architecture/optimizer: https://github.com/bigscience-workshop/bigscience/tree/master/train/tr11-176B-ml

Blog post on the hardware/engineering side: https://bigscience.huggingface.co/blog/which-hardware-to-train-a-176b-parameters-model

Details on the distributed setup used for the training: https://github.com/bigscience-workshop/bigscience/tree/master/train/tr11-176B-ml

Tensorboard updated during the training: https://huggingface.co/bigscience/tr11-176B-ml-logs/tensorboard#scalars&tagFilter=loss

## Lessons

Insights on how to approach training, negative results: https://github.com/bigscience-workshop/bigscience/blob/master/train/lessons-learned.md

Details on the obstacles overcome during the preparation on the engineering side (instabilities, optimization of training throughput, so many technical tricks and questions): https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md

## Initial Results

Initial prompting experiments using interim checkpoints: https://huggingface.co/spaces/bigscience/bloom-book

</details>


## Original checkpoints

The checkpoints in this repo correspond to the HuggingFace Transformers format. If you want to use our fork of [Megatron-DeepSpeed](https://github.com/bigscience-workshop/Megatron-DeepSpeed) that the model was trained with, you'd want to use [this repo instead](https://huggingface.co/bigscience/bloom-optimizer-states).

Many intermediate checkpoints are available at https://huggingface.co/bigscience/bloom-intermediate/

---

# Model Card Authors
*Ordered roughly chronologically and by amount of time spent on creating this model card.*

Margaret Mitchell, Giada Pistilli, Yacine Jernite, Ezinwanne Ozoani, Marissa Gerchick, Nazneen Rajani, Sasha Luccioni, Irene Solaiman, Maraim Masoud, Somaieh Nikpoor, Carlos Muñoz Ferrandis, Stas Bekman, Christopher Akiki, Danish Contractor, David Lansky, Angelina McMillan-Major, Tristan Thrush, Suzana Ilić, Gérard Dupont, Shayne Longpre, Manan Dey, Stella Biderman, Douwe Kiela, Emi Baylor, Teven Le Scao, Aaron Gokaslan, Julien Launay, Niklas Muennighoff