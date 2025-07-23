## Model Details
This section provides fundamental information about the model, helping stakeholders understand its context and key characteristics.

### Person or organization developing model:
C V Theodoris, L Xiao, A Chopra, M D Chaffin, Z R Al Sayed, M C Hill, H Mantineo, E Brydon, Z Zeng, X S Liu, P T Ellinor, H Chen, M S Venkatesh, J Gomez Ortega, S V Mahesh, T Nandi, R Madduri, K Pelka.

### Model date:
The model was developed with key milestones in June 2021 (original model trained on ~30 million transcriptomes) and April 2024 (expanded model trained on ~95 million transcriptomes).

### Model version:
The repository includes several pretrained models:
- GF-6L-30M-i2048 (June 2021)
- GF-12L-30M-i2048 (June 2021)
- GF-12L-95M-i4096 (April 2024)
- GF-20L-95M-i4096 (April 2024)

The current default model in the main directory of the repository is GF-12L-95M-i4096.

Additionally, there is a cancer-tuned model: GF-12L-95M-i4096_CLcancer, which was created following continual learning on ~14 million cancer cells.

The version differences are primarily in the number of layers (L), the size of the pretraining dataset (M=millions of cells), and the input size (i), as well as the pretraining date. Some models are also fine-tuned for specific domains like cancer.

### Model type:
Geneformer is a foundational transformer model. It utilizes transformer encoder units with N layers, where N varies depending on the model size (e.g., 6L, 12L, 20L indicating 6, 12, and 20 layers respectively). The input to the model is a rank value encoding of a single cell’s transcriptome. Pretraining is based on a masked learning objective.

### Training details:
Geneformer was pretrained using a self-supervised masked learning objective. 15% of the genes within each transcriptome were masked, and the model was trained to predict which gene should be within each masked position in that specific cell state using the context of the remaining unmasked genes. The training data consisted of a large-scale corpus of single cell transcriptomes. Initially, the model was pretrained on ~30 million single cell transcriptomes (Genecorpus-30M) in June 2021, excluding cells with high mutational burdens (e.g. malignant cells and immortalized cell lines). Later, in April 2024, an expanded model was pretrained on ~95 million non-cancer transcriptomes, followed by continual learning on ~14 million cancer transcriptomes to create a cancer domain-tuned model.

### Paper or other resource for more information:
- [our manuscript](https://rdcu.be/ddrx0) for details of the original model trained on ~30 million transcriptomes in June 2021 and the initial report of our in silico perturbation and cell and gene classification strategies.
- [our manuscript](https://www.biorxiv.org/content/10.1101/2024.08.16.608180v1.full.pdf) for details of the expanded model trained on ~95 million transcriptomes in April 2024 and our continual learning, multitask learning, and quantization strategies.
- [geneformer.readthedocs.io](https://geneformer.readthedocs.io) for documentation.

### Citation details:
- C V Theodoris#, L Xiao, A Chopra, M D Chaffin, Z R Al Sayed, M C Hill, H Mantineo, E Brydon, Z Zeng, X S Liu, P T Ellinor#. Transfer learning enables predictions in network biology. _**Nature**_, 31 May 2023. (#co-corresponding authors)
- H Chen*, M S Venkatesh*, J Gomez Ortega, S V Mahesh, T Nandi, R Madduri, K Pelka†, C V Theodoris†#. Quantized multi-task learning for context-specific representations of gene network dynamics. _**bioRxiv**_, 19 Aug 2024. (*co-first authors, †co-senior authors, #corresponding author)

### License:
Not available.

### Contact:
Not available.

---

## Intended Use
This section outlines the intended applications of the model.

### Primary intended uses:
The primary intended use of Geneformer is for context-aware predictions in network biology, particularly in settings with limited data. It can be used directly for zero-shot learning, for example for in silico perturbation analysis or fine-tuned for relevant downstream tasks, such as gene or cell state classifications. Example applications demonstrated in [our manuscript](https://rdcu.be/ddrx0) include:

*Fine-tuning*:
- transcription factor dosage sensitivity
- chromatin dynamics (bivalently marked promoters)
- transcription factor regulatory range
- gene network centrality
- transcription factor targets
- cell type annotation
- batch integration
- cell state classification across differentiation
- disease classification
- in silico perturbation to determine disease-driving genes
- in silico treatment to determine candidate therapeutic targets

*Zero-shot learning*:
- batch integration
- gene context specificity
- in silico reprogramming
- in silico differentiation
- in silico perturbation to determine impact on cell state
- in silico perturbation to determine transcription factor targets
- in silico perturbation to determine transcription factor cooperativity

The model is designed to understand gene network dynamics and can be applied to accelerate the discovery of key network regulators and candidate therapeutic targets. The input is a single cell’s transcriptome, and the output depends on the specific application.

### Primary intended users:
Users in network biology, and related fields who aim to understand gene network dynamics, discover therapeutic targets, and perform in silico experiments.

### Out-of-scope uses:
Not available.

---

## How to Use
This section outlines how to use the model.

To install Geneformer and access its functionalities:

```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/ctheodoris/Geneformer
cd Geneformer
pip install .
```

For usage examples, refer to the [examples directory](https://huggingface.co/ctheodoris/Geneformer/tree/main/examples) which includes scripts for:
- tokenizing transcriptomes
- pretraining
- hyperparameter tuning
- fine-tuning
- extracting and plotting cell embeddings
- in silico perturbation

The repository also contains pretrained models and fine-tuned models in respective directories. The default pretrained model in the main directory of the repository is GF-12L-95M-i4096. Fine-tuned models are located in the `fine_tuned_models` directory, and the cancer-tuned model is GF-12L-95M-i4096_CLcancer.

Please note that fine-tuning examples are general, and users need to adapt input datasets and labels based on their specific downstream tasks. Example input files for some downstream tasks are available in the [example_input_files directory](https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/example_input_files) in the dataset repository.

---

## Factors
This section addresses variables that may impact the model's performance.

### Relevant factors:
Efficient usage of Geneformer requires GPU resources. Hyperparameter tuning is strongly recommended for each downstream fine-tuning application to significantly boost predictive potential. Key hyperparameters include max learning rate, learning schedule, and the number of layers to freeze during fine-tuning.

### Evaluation factors:
Not available.

---

## Metrics
This section describes how the model's performance is evaluated.

### Model performance measures:
Not available.

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
Geneformer was trained on:
- [Genecorpus-30M](https://huggingface.co/datasets/ctheodoris/Genecorpus-30M): A corpus of ~30 million single cell transcriptomes used for the original pretraining in June 2021.
- A corpus of ~95 million non-cancer transcriptomes used for pretraining in April 2024.
- A corpus of ~14 million cancer transcriptomes used for continual learning to create the cancer-tuned model.

These datasets are large-scale collections of single-cell transcriptomic data. Genecorpus-30M is available on Hugging Face.

### Motivation:
The datasets were chosen to provide a large and diverse representation of human single cell transcriptomes, covering a broad range of human tissues. The large scale of the data enables self-supervised learning and the capture of complex gene network dynamics without relying on labeled data allows the inclusion of such large amount of data. Excluding cells with high mutational burdens (initially) aimed to facilitate interpretation by preventing substantial network rewiring in the absence of companion genome sequencing. The inclusion of cancer transcriptomes for continual learning allows for domain-specific tuning for cancer-related applications.

### Preprocessing:
Each single cell’s transcriptome was preprocessed using rank value encoding. Genes are ranked by their expression in each cell, scaled by their expression across the entire Genecorpus-30M corpus. This method prioritizes genes that distinguish cell state and may be more robust against technical artifacts that bias absolute transcript counts.

---

## Quantitative Analyses
This section presents disaggregated evaluation results.

### Unitary results:
Not available.

### Intersectional results:
Not available.

---

## Memory or Hardware Requirements
This section outlines the memory or hardware requirements for loading, deploying, and training the model.

### Loading Requirements:
GPU is recommended.

### Deploying Requirements:
Not available.

### Training or Fine-tuning Requirements:
Not available.

---

## Ethical Considerations
This section discusses the ethical considerations in model development, including challenges, risks, and solutions.

Not available.

---

## Caveats and Recommendations
This section lists unresolved issues and provides guidance for users.

### Caveats:
For optimal performance in downstream tasks, hyperparameter tuning is crucial. The provided fine-tuning examples are meant to be generally applicable, and users need to adapt them to their specific datasets and tasks.

### Recommendations:
It is strongly recommended to tune hyperparameters for each downstream fine-tuning application to maximize predictive potential. Users should consider tuning parameters such as max learning rate, learning schedule, and the number of layers to freeze. For fine-tuning, users should adapt the provided examples to their specific downstream tasks and datasets.

## Additional Information

# Model Description
The rank value encoding provides a nonparametric representation of that cell’s transcriptome and takes advantage of the many observations of each gene’s expression across the pretraining corpus to prioritize genes that distinguish cell state. Specifically, this method will deprioritize ubiquitously highly-expressed housekeeping genes by scaling them to a lower rank. Conversely, genes such as transcription factors that may be lowly expressed when they are expressed but highly distinguish cell state will move to a higher rank within the encoding. Furthermore, this rank-based approach may be more robust against technical artifacts that may systematically bias the absolute transcript counts value while the overall relative ranking of genes within each cell remains more stable.

During pretraining, Geneformer gained a fundamental understanding of network dynamics, encoding network hierarchy in the model’s attention weights in a completely self-supervised manner. With both zero-shot learning and fine-tuning with limited task-specific data, Geneformer consistently boosted predictive accuracy in a diverse panel of downstream tasks relevant to chromatin and network dynamics. In silico perturbation with zero-shot learning identified a novel transcription factor in cardiomyocytes that we experimentally validated to be critical to their ability to generate contractile force. In silico treatment with limited patient data revealed candidate therapeutic targets for cardiomyopathy that we experimentally validated to significantly improve the ability of cardiomyocytes to generate contractile force in an induced pluripotent stem cell (iPSC) model of the disease. Overall, Geneformer represents a foundational deep learning model pretrained on a large-scale corpus human single cell transcriptomes to gain a fundamental understanding of gene network dynamics that can now be democratized to a vast array of downstream tasks to accelerate discovery of key network regulators and candidate therapeutic targets.

# Installation
In addition to the pretrained model, contained herein are functions for tokenizing and collating data specific to single cell transcriptomics, pretraining the model, fine-tuning the model, extracting and plotting cell embeddings, and performing in silico pertrubation with either the pretrained or fine-tuned models.

Example input files located within the [example_input_files directory](https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/example_input_files) in the dataset repository only represent a few example fine-tuning applications.