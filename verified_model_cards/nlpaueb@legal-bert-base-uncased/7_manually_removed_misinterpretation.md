## Model Details
This section provides fundamental information about the model, helping stakeholders understand its context and key characteristics.

### Person or organization developing model:
[AUEB's Natural Language Processing Group](http://nlp.cs.aueb.gr) develops algorithms, models, and systems that allow computers to process and generate natural language texts.

The group's current research interests include:
* question answering systems for databases, ontologies, document collections, and the Web, especially biomedical question answering,
* natural language generation from databases and ontologies, especially Semantic Web ontologies,
text classification, including filtering spam and abusive content,
* information extraction and opinion mining, including legal text analytics and sentiment analysis,
* natural language processing tools for Greek, for example parsers and named-entity recognizers,
machine learning in natural language processing, especially deep learning.

The group is part of the Information Processing Laboratory of the Department of Informatics of the Athens University of Economics and Business.

[Ilias Chalkidis](https://iliaschalkidis.github.io) on behalf of [AUEB's Natural Language Processing Group](http://nlp.cs.aueb.gr)

| Github: [@ilias.chalkidis](https://github.com/iliaschalkidis) | Twitter: [@KiddoThe2B](https://twitter.com/KiddoThe2B) |

### Model date:
2020. Publication in Findings of Empirical Methods in Natural Language Processing (EMNLP 2020).

### Model version:
LEGAL-BERT is a family of BERT models for the legal domain, including sub-domain variants (CONTRACTS-, EURLEX-, ECHR-) and general LEGAL-BERT, as well as light-weight model (LEGAL-BERT-SMALL).
\* LEGAL-BERT-BASE is the model referred to as LEGAL-BERT-SC in Chalkidis et al. (2020); a model trained from scratch in the legal corpora using a newly created vocabulary by a sentence-piece tokenizer trained on the very same corpora.

\*\* As many of you expressed interest in the LEGAL-BERT-FP models (those relying on the original BERT-BASE checkpoint), they have been released in Archive.org (https://archive.org/details/legal_bert_fp), as these models are secondary and possibly only interesting for those who aim to dig deeper in the open questions of Chalkidis et al. (2020).

### Model type:
LEGAL-BERT is a family of BERT models for the legal domain.
We released a model similar to the English BERT-BASE model (12-layer, 768-hidden, 12-heads, 110M parameters).
Part of LEGAL-BERT is a light-weight model pre-trained from scratch on legal data, which achieves comparable performance to larger models, while being much more efficient (approximately 4 times faster) with a smaller environmental footprint (33% the size of BERT-BASE).

### Training details:
We trained BERT using the official code provided in Google BERT's GitHub repository (https://github.com/google-research/bert).
We chose to follow the same training set-up: 1 million training steps with batches of 256 sequences of length 512 with an initial learning rate 1e-4.
We were able to use a single Google Cloud TPU v3-8 provided for free from [TensorFlow Research Cloud (TFRC)](https://www.tensorflow.org/tfrc), while also utilizing [GCP research credits](https://edu.google.com/programs/credits/research). Huge thanks to both Google programs for supporting us!

### Paper or other resource for more information:
I. Chalkidis, M. Fergadiotis, P. Malakasiotis, N. Aletras and I. Androutsopoulos. "LEGAL-BERT: The Muppets straight out of Law School". In Findings of Empirical Methods in Natural Language Processing (EMNLP 2020) (Short Papers), to be held online, 2020. (https://aclanthology.org/2020.findings-emnlp.261)

Google BERT's GitHub repository: (https://github.com/google-research/bert)

TensorFlow Research Cloud (TFRC): (https://www.tensorflow.org/tfrc)

GCP research credits: (https://edu.google.com/programs/credits/research)

LEGAL-BERT-FP models in Archive.org: (https://archive.org/details/legal_bert_fp)

### Citation details:
```
@inproceedings{chalkidis-etal-2020-legal,
    title = "{LEGAL}-{BERT}: The Muppets straight out of Law School",
    author = "Chalkidis, Ilias  and
      Fergadiotis, Manos  and
      Malakasiotis, Prodromos  and
      Aletras, Nikolaos  and
      Androutsopoulos, Ion",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    doi = "10.18653/v1/2020.findings-emnlp.261",
    pages = "2898--2904"
}
```

### License:
Not available.

### Contact:
Not available.

---

## Intended Use
This section outlines the intended applications of the model.

### Primary intended uses:
LEGAL-BERT is intended to assist legal NLP research, computational law, and legal technology applications. Sub-domain variants (CONTRACTS-, EURLEX-, ECHR-) and/or general LEGAL-BERT perform better than using BERT out of the box for domain-specific tasks.

### Primary intended users:
Legal NLP research community, computational law researchers, and legal technology developers.

### Out-of-scope uses:
Not available.

---

## How to Use
This section outlines how to use the model.

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
```

| Corpus                             | Model                               | Masked token | Predictions  |
| --------------------------------- | ---------------------------------- | ------------ | ------------ |
|  | **BERT-BASE-UNCASED**                 |
| (Contracts) | This [MASK] Agreement is between General Motors and John Murray . | employment | ('new', '0.09'), ('current', '0.04'), ('proposed', '0.03'), ('marketing', '0.03'), ('joint', '0.02')
| (ECHR) | The applicant submitted that her husband was subjected to treatment amounting to [MASK] whilst in the custody of Adana Security Directorate | torture | ('torture', '0.32'), ('rape', '0.22'), ('abuse', '0.14'), ('death', '0.04'), ('violence', '0.03')
| (EURLEX) | Establishing a system for the identification and registration of [MASK] animals and regarding the labelling of beef and beef products . | bovine | ('farm', '0.25'), ('livestock', '0.08'), ('draft', '0.06'), ('domestic', '0.05'), ('wild', '0.05')
|  | **CONTRACTS-BERT-BASE**                 |
| (Contracts) | This [MASK] Agreement is between General Motors and John Murray . | employment | ('letter', '0.38'), ('dealer', '0.04'), ('employment', '0.03'), ('award', '0.03'), ('contribution', '0.02')
| (ECHR) | The applicant submitted that her husband was subjected to treatment amounting to [MASK] whilst in the custody of Adana Security Directorate | torture | ('death', '0.39'), ('imprisonment', '0.07'), ('contempt', '0.05'), ('being', '0.03'), ('crime', '0.02')
| (EURLEX) | Establishing a system for the identification and registration of [MASK] animals and regarding the labelling of beef and beef products . | bovine | (('domestic', '0.18'), ('laboratory', '0.07'), ('household', '0.06'), ('personal', '0.06'), ('the', '0.04')
|  | **EURLEX-BERT-BASE**                 |
| (Contracts) | This [MASK] Agreement is between General Motors and John Murray . | employment | ('supply', '0.11'), ('cooperation', '0.08'), ('service', '0.07'), ('licence', '0.07'), ('distribution', '0.05')
| (ECHR) | The applicant submitted that her husband was subjected to treatment amounting to [MASK] whilst in the custody of Adana Security Directorate | torture | ('torture', '0.66'), ('death', '0.07'), ('imprisonment', '0.07'), ('murder', '0.04'), ('rape', '0.02')
| (EURLEX) | Establishing a system for the identification and registration of [MASK] animals and regarding the labelling of beef and beef products . | bovine | ('live', '0.43'), ('pet', '0.28'), ('certain', '0.05'), ('fur', '0.03'), ('the', '0.02')
|  | **ECHR-BERT-BASE**                 |
| (Contracts) | This [MASK] Agreement is between General Motors and John Murray . | employment | ('second', '0.24'), ('latter', '0.10'), ('draft', '0.05'), ('bilateral', '0.05'), ('arbitration', '0.04')
| (ECHR) | The applicant submitted that her husband was subjected to treatment amounting to [MASK] whilst in the custody of Adana Security Directorate | torture | ('torture', '0.99'), ('death', '0.01'), ('inhuman', '0.00'), ('beating', '0.00'), ('rape', '0.00')
| (EURLEX) | Establishing a system for the identification and registration of [MASK] animals and regarding the labelling of beef and beef products . | bovine | ('pet', '0.17'), ('all', '0.12'), ('slaughtered', '0.10'), ('domestic', '0.07'), ('individual', '0.05')
|  | **LEGAL-BERT-BASE**                |
| (Contracts) | This [MASK] Agreement is between General Motors and John Murray . | employment | ('settlement', '0.26'), ('letter', '0.23'), ('dealer', '0.04'), ('master', '0.02'), ('supplemental', '0.02')
| (ECHR) | The applicant submitted that her husband was subjected to treatment amounting to [MASK] whilst in the custody of Adana Security Directorate | torture | ('torture', '1.00'), ('detention', '0.00'), ('arrest', '0.00'), ('rape', '0.00'), ('death', '0.00')
| (EURLEX) | Establishing a system for the identification and registration of [MASK] animals and regarding the labelling of beef and beef products . | bovine | ('live', '0.67'), ('beef', '0.17'), ('farm', '0.03'), ('pet', '0.02'), ('dairy', '0.01')
|  | **LEGAL-BERT-SMALL**                |
| (Contracts) | This [MASK] Agreement is between General Motors and John Murray . | employment | ('license', '0.09'), ('transition', '0.08'), ('settlement', '0.04'), ('consent', '0.03'), ('letter', '0.03')
| (ECHR) | The applicant submitted that her husband was subjected to treatment amounting to [MASK] whilst in the custody of Adana Security Directorate | torture | ('torture', '0.59'), ('pain', '0.05'), ('ptsd', '0.05'), ('death', '0.02'), ('tuberculosis', '0.02')
| (EURLEX) | Establishing a system for the identification and registration of [MASK] animals and regarding the labelling of beef and beef products . | bovine | ('all', '0.08'), ('live', '0.07'), ('certain', '0.07'), ('the', '0.07'), ('farm', '0.05')

---

## Factors
This section addresses variables that may impact the model's performance.

### Relevant factors:
Legal domain. Performance of LEGAL-BERT models is domain-specific and intended for legal text.

### Evaluation factors:
Consider the experiments in the article "LEGAL-BERT: The Muppets straight out of Law School". Chalkidis et al., 2020, (https://aclanthology.org/2020.findings-emnlp.261)

---

## Metrics
This section describes how the model's performance is evaluated.

### Model performance measures:
Evaluation on downstream tasks is mentioned in the paper "LEGAL-BERT: The Muppets straight out of Law School". Chalkidis et al., 2020, (https://aclanthology.org/2020.findings-emnlp.261)

### Decision thresholds:
Not available.

### Variation approaches:
Not available.

---

## Evaluation Data
This section provides details about the datasets used to evaluate the model.

### Datasets:
Evaluation datasets are discussed in the paper "LEGAL-BERT: The Muppets straight out of Law School". Chalkidis et al., 2020, (https://aclanthology.org/2020.findings-emnlp.261)

### Motivation:
Not available.

### Preprocessing:
Not available.

---

## Training Data
This section provides details about the datasets used to train the model.

### Datasets:
The pre-training corpora of LEGAL-BERT include:

* 116,062 documents of EU legislation, publicly available from EURLEX (http://eur-lex.europa.eu), the repository of EU Law running under the EU Publication Office.

* 61,826 documents of UK legislation, publicly available from the UK legislation portal (http://www.legislation.gov.uk).

* 19,867 cases from the European Court of Justice (ECJ), also available from EURLEX.

* 12,554 cases from HUDOC, the repository of the European Court of Human Rights (ECHR) (http://hudoc.echr.coe.int/eng).

* 164,141 cases from various courts across the USA, hosted in the Case Law Access Project portal (https://case.law).

* 76,366 US contracts from EDGAR, the database of US Securities and Exchange Commission (SECOM) (https://www.sec.gov/edgar.shtml).

### Motivation:
To pre-train the different variations of LEGAL-BERT, we collected 12 GB of diverse English legal text from several fields (e.g., legislation, court cases, contracts) scraped from publicly available resources.

### Preprocessing:
We released a model trained from scratch in the legal corpora using a newly created vocabulary by a sentence-piece tokenizer trained on the very same corpora.

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
Not available.

### Deploying Requirements:
Part of LEGAL-BERT is a light-weight model pre-trained from scratch on legal data, which achieves comparable performance to larger models, while being much more efficient (approximately 4 times faster) with a smaller environmental footprint.

### Training or Fine-tuning Requirements:
We were able to use a single Google Cloud TPU v3-8 provided for free from [TensorFlow Research Cloud (TFRC)](https://www.tensorflow.org/tfrc).

---

## Ethical Considerations
This section discusses the ethical considerations in model development, including challenges, risks, and solutions.

Not available.

---

## Caveats and Recommendations
This section lists unresolved issues and provides guidance for users.

### Caveats:
LEGAL-BERT-FP models are secondary and possibly only interesting for those who aim to dig deeper in the open questions of Chalkidis et al. (2020).

### Recommendations:
Sub-domain variants (CONTRACTS-, EURLEX-, ECHR-) and/or general LEGAL-BERT perform better than using BERT out of the box for domain-specific tasks.

---

## Additional Information

# LEGAL-BERT: The Muppets straight out of Law School

<img align="left" src="https://i.ibb.co/p3kQ7Rw/Screenshot-2020-10-06-at-12-16-36-PM.png" width="100"/>

LEGAL-BERT is a family of BERT models for the legal domain, intended to assist legal NLP research, computational law, and legal technology applications.
<br/><br/>

---

## Models list

| Model name          | Model Path                            | Training corpora    |
| ------------------- | ------------------------------------  | ------------------- |
| CONTRACTS-BERT-BASE | `nlpaueb/bert-base-uncased-contracts` | US contracts        |
| EURLEX-BERT-BASE    | `nlpaueb/bert-base-uncased-eurlex`    | EU legislation      |
| ECHR-BERT-BASE      | `nlpaueb/bert-base-uncased-echr`      | ECHR cases          |
| LEGAL-BERT-BASE *     | `nlpaueb/legal-bert-base-uncased`     | All                 |
| LEGAL-BERT-SMALL    | `nlpaueb/legal-bert-small-uncased`    | All                 |