# Geneformer
- Geneformer is a foundational transformer model.
- Geneformer is pretrained on a large-scale corpus of single cell transcriptomes.
- Geneformer enables context-aware predictions in settings with limited data in network biology.
- See [our manuscript](https://rdcu.be/ddrx0) for details of the original model trained on ~30 million transcriptomes in June 2021.
- The initial report includes in silico perturbation and cell and gene classification strategies.
- See [our manuscript](https://www.biorxiv.org/content/10.1101/2024.08.16.608180v1.full.pdf) for details of the expanded model trained on ~95 million transcriptomes in April 2024.
- The expanded model includes continual learning, multitask learning, and quantization strategies.
- See [geneformer.readthedocs.io](https://geneformer.readthedocs.io) for documentation.

# Model Description
- Geneformer is pretrained on a large-scale corpus of single cell transcriptomes representing a broad range of human tissues.
- Geneformer was originally pretrained in June 2021 on [Genecorpus-30M](https://huggingface.co/datasets/ctheodoris/Genecorpus-30M).
- Genecorpus-30M is a corpus comprised of ~30 million single cell transcriptomes.
- Cells with high mutational burdens (e.g. malignant cells and immortalized cell lines) were excluded.
- Exclusion of these cells prevents substantial network rewiring without companion genome sequencing.
- The exclusion facilitate interpretation.
- In April 2024, Geneformer was pretrained on ~95 million non-cancer transcriptomes.
- Geneformer underwent continual learning on ~14 million cancer transcriptomes to yield a cancer domain-tuned model.
- Each single cell’s transcriptome is presented to the model as a rank value encoding.
- Genes are ranked by their expression in that cell scaled by their expression across the entire Genecorpus-30M.
- The rank value encoding provides a nonparametric representation of that cell’s transcriptome.
- This encoding takes advantage of many observations of each gene’s expression across the pretraining corpus.
- The method prioritizes genes that distinguish cell state.
- Ubiquitously highly-expressed housekeeping genes are deprioritized by scaling them to a lower rank.
- Genes such as transcription factors that may be lowly expressed but highly distinguish cell state will move to a higher rank.
- The rank-based approach may be more robust against technical artifacts that bias absolute transcript counts.
- The overall relative ranking of genes within each cell remains more stable.
- The rank value encoding of each single cell’s transcriptome proceeds through N layers of transformer encoder units.
- N varies depending on the model size.
- Pretraining was accomplished using a masked learning objective.
- 15% of the genes within each transcriptome were masked.
- The model was trained to predict which gene should be in each masked position using the context of the remaining unmasked genes.
- This approach is entirely self-supervised.
- The model can be trained on completely unlabeled data.
- This allows the inclusion of large amounts of training data without being restricted to samples with accompanying labels.
- Applications and results are detailed in [our manuscript](https://rdcu.be/ddrx0).
- During pretraining, Geneformer gained a fundamental understanding of network dynamics by encoding network hierarchy in the model’s attention weights in a completely self-supervised manner.
- Geneformer boosts predictive accuracy in a diverse panel of downstream tasks relevant to chromatin and network dynamics with both zero-shot learning and fine-tuning with limited task-specific data.
- In silico perturbation with zero-shot learning identified a novel transcription factor in cardiomyocytes.
- The novel transcription factor was experimentally validated to be critical to cardiomyocytes' ability to generate contractile force.
- In silico treatment with limited patient data revealed candidate therapeutic targets for cardiomyopathy.
- The candidate therapeutic targets were experimentally validated to significantly improve the ability of cardiomyocytes to generate contractile force in an induced pluripotent stem cell (iPSC) model of the disease.
- Geneformer represents a foundational deep learning model pretrained on a large-scale corpus of human single cell transcriptomes.
- Geneformer gains a fundamental understanding of gene network dynamics.
- Geneformer can be democratized to a vast array of downstream tasks to accelerate discovery of key network regulators and candidate therapeutic targets.

## Pretrained Models
- The repository includes the following pretrained models:
  - GF-6L-30M-i2048 (June 2021)
  - GF-12L-30M-i2048 (June 2021)
  - GF-12L-95M-i4096 (April 2024)
  - GF-20L-95M-i4096 (April 2024)
  - Meaning of the symbols and abbreviations:
    L=layers
    M=millions of cells used for pretraining
    i=input size
    (pretraining date)
- The current default model in the main directory of the repository is GF-12L-95M-i4096.
- The repository contains fine-tuned models in the fine_tuned_models directory.
- The cancer-tuned model following continual learning on ~14 million cancer cells is GF-12L-95M-i4096_CLcancer.

# Application
- The pretrained Geneformer model can be used directly for zero-shot learning.
- The pretrained Geneformer model can be used for in silico perturbation analysis.
- The pretrained Geneformer model can be fine-tuned towards the relevant downstream task such as gene or cell state classification.
- Example applications demonstrated in [our manuscript](https://rdcu.be/ddrx0) include:
  - *Fine-tuning*:
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
  - *Zero-shot learning*:
    - batch integration
    - gene context specificity
    - in silico reprogramming
    - in silico differentiation
    - in silico perturbation to determine impact on cell state
    - in silico perturbation to determine transcription factor targets
    - in silico perturbation to determine transcription factor cooperativity

# Installation
- The repository contains functions for:
  - tokenizing and collating data specific to single cell transcriptomics
  - pretraining the model
  - fine-tuning the model
  - extracting and plotting cell embeddings
  - performing in silico perturbation with either the pretrained or fine-tuned models.
- To install:
  ```bash
  # Make sure you have git-lfs installed (https://git-lfs.com)
  git lfs install
  git clone https://huggingface.co/ctheodoris/Geneformer
  cd Geneformer
  pip install .
  ```
- For usage, see [examples](https://huggingface.co/ctheodoris/Geneformer/tree/main/examples) for:
  - tokenizing transcriptomes
  - pretraining
  - hyperparameter tuning
  - fine-tuning
  - extracting and plotting cell embeddings
  - in silico perturbation
- Fine-tuning examples are meant to be generally applicable.
- The input datasets and labels will vary depending on the downstream task.
- Example input files for a few downstream tasks are located within the [example_input_files directory](https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/example_input_files) in the dataset repository.
- These example input files only represent a few example fine-tuning applications.
- GPU resources are required for efficient usage of Geneformer.
- It is strongly recommended to tune hyperparameters for each downstream fine-tuning application.
- Tuning hyperparameters can significantly boost predictive potential in the downstream task (e.g. max learning rate, learning schedule, number of layers to freeze, etc.).

# Citations
- C V Theodoris#, L Xiao, A Chopra, M D Chaffin, Z R Al Sayed, M C Hill, H Mantineo, E Brydon, Z Zeng, X S Liu, P T Ellinor#. Transfer learning enables predictions in network biology. _**Nature**_, 31 May 2023. (#co-corresponding authors)
- H Chen*, M S Venkatesh*, J Gomez Ortega, S V Mahesh, T Nandi, R Madduri, K Pelka†, C V Theodoris†#. Quantized multi-task learning for context-specific representations of gene network dynamics. _**bioRxiv**_, 19 Aug 2024. (*co-first authors, †co-senior authors, #corresponding author)