## Model Details
This section provides fundamental information about the model, helping stakeholders understand its context and key characteristics.

### Person or organization developing model:
TinyTimeMixers (TTMs) are open-sourced by IBM Research. IBM developers produced this code as an open source project. Model Card Authors: Vijay Ekambaram, Arindam Jati, Pankaj Dayama, Nam H. Nguyen, Wesley Gifford and Jayant Kalagnanam.

### Model date:
Recent updates indicate ongoing development and refinement. The paper associated with the model was published in January 2024.

### Model version:
The current open-source version is referred to as TTM-1.  There are also newer variants: TTM-B, TTM-E and TTM-A. This model card includes the 512-96 and 1024-96 versions, with the 1024-96 model having 1M parameters. Each pre-trained model is released in a different branch name.

### Model type:
TinyTimeMixer (TTM) is a compact pre-trained model for Multivariate Time-Series Forecasting. It is categorized as a "focused pre-trained model", tailored for specific forecasting settings based on context length and forecast length. The architecture details are available in the paper.

### Training details:
TTMs are pre-trained on publicly available time series data with various augmentations. Pretraining TTMs is fast, taking only 3-6 hours using 6 A100 GPUs.

### Paper or other resource for more information:
- [paper](https://arxiv.org/pdf/2401.03955v5.pdf)
- [paper (Newer variants, extended benchmarks)](https://arxiv.org/pdf/2401.03955.pdf)
- **Repository:** https://github.com/IBM/tsfm/tree/main/tsfm_public/models/tinytimemixer
- [colab](https://colab.research.google.com/github/IBM/tsfm/blob/main/notebooks/tutorial/ttm_tutorial.ipynb)
- [Getting Started Notebook](https://github.com/IBM/tsfm/blob/main/notebooks/hfdemo/ttm_getting_started.ipynb)
- [512-96 Benchmarks](https://github.com/IBM/tsfm/blob/main/notebooks/hfdemo/tinytimemixer/ttm_benchmarking_512_96.ipynb)
- [1024-96 Benchmarks](https://github.com/IBM/tsfm/blob/main/notebooks/hfdemo/tinytimemixer/ttm_benchmarking_1024_96.ipynb)
- [notebook]](https://github.com/IBM/tsfm/blob/main/notebooks/hfdemo/tinytimemixer/ttm_benchmarking_1024_96.ipynb)
- [notebook]](https://github.com/IBM/tsfm/blob/main/notebooks/hfdemo/tinytimemixer/ttm_m4_hourly.ipynb)
- **External Blogs on TTM**:
    - https://aihorizonforecast.substack.com/p/tiny-time-mixersttms-powerful-zerofew
    - https://medium.com/@david.proietti_17/predicting-venetian-lagoon-tide-levels-with-multivariate-time-series-modeling-8bafdf229588

### Citation details:
**BibTeX:**

```
@misc{ekambaram2024tiny,
      title={Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series},
      author={Vijay Ekambaram and Arindam Jati and Pankaj Dayama and Sumanta Mukherjee and Nam H. Nguyen and Wesley M. Gifford and Chandra Reddy and Jayant Kalagnanam},
      year={2024},
      eprint={2401.03955},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

**APA:**

Ekambaram, V., Jati, A., Dayama, P., Mukherjee, S., Nguyen, N. H., Gifford, W. M., … Kalagnanam, J. (2024). Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series. arXiv [Cs.LG]. Retrieved from http://arxiv.org/abs/2401.03955

### License:
The model is provided under an open source software license. All content in the repository is provided by IBM under the associated open source software license.

### Contact:
Model Card Authors: Vijay Ekambaram, Arindam Jati, Pankaj Dayama, Nam H. Nguyen, Wesley Gifford and Jayant Kalagnanam.

---

## Intended Use
This section outlines the intended applications of the model.

### Primary intended uses:
TTM is primarily intended for Multivariate Time-Series Forecasting, specifically for point forecasting use-cases ranging from minutely to hourly resolutions (Ex. 10 min, 15 min, 1 hour.). It excels in zero-shot and few-shot forecasting scenarios. TTM-1 currently supports 2 modes: Zeroshot forecasting and Finetuned forecasting. It supports multivariate forecasting via both channel independence and channel-mixing approaches.

### Primary intended users:
Users interested in time-series forecasting.

### Out-of-scope uses:
TTM is not intended for use with lower resolutions such as weekly or monthly, as it requires a minimum context length of 512 or 1024. Enabling upsampling or prepending zeros to increase context length for shorter datasets is not recommended.

---

## How to Use
This section outlines how to use the model.

- [colab](https://colab.research.google.com/github/IBM/tsfm/blob/main/notebooks/tutorial/ttm_tutorial.ipynb)
- [Getting Started Notebook](https://github.com/IBM/tsfm/blob/main/notebooks/hfdemo/ttm_getting_started.ipynb)
- [512-96 Benchmarks](https://github.com/IBM/tsfm/blob/main/notebooks/hfdemo/tinytimemixer/ttm_benchmarking_512_96.ipynb)
- [1024-96 Benchmarks](https://github.com/IBM/tsfm/blob/main/notebooks/hfdemo/tinytimemixer/ttm_benchmarking_1024_96.ipynb)
- Script for Finetuning with cross-channel correlation support - to be added soon

**Code Snippets:**

```
# Load Model from HF Model Hub mentioning the branch name in revision field

model = TinyTimeMixerForPrediction.from_pretrained(
                "https://huggingface.co/ibm/TTM", revision="main"
            )

# Do zeroshot
zeroshot_trainer = Trainer(
        model=model,
        args=zeroshot_forecast_args,
        )
    )

zeroshot_output = zeroshot_trainer.evaluate(dset_test)


# Freeze backbone and enable few-shot or finetuning:

# freeze backbone
for param in model.backbone.parameters():
  param.requires_grad = False

finetune_forecast_trainer = Trainer(
        model=model,
        args=finetune_forecast_args,
        train_dataset=dset_train,
        eval_dataset=dset_val,
        callbacks=[early_stopping_callback, tracking_callback],
        optimizers=(optimizer, scheduler),
    )
finetune_forecast_trainer.train()
fewshot_output = finetune_forecast_trainer.evaluate(dset_test)
```

**Recommended Use:**
1. Users have to externally standard scale their data independently for every channel before feeding it to the model (Refer to [TSP](https://github.com/IBM/tsfm/blob/main/tsfm_public/toolkit/time_series_preprocessor.py), our data processing utility for data scaling.)
2. The current open-source version supports only minutely and hourly resolutions(Ex. 10 min, 15 min, 1 hour.). Other lower resolutions (say weekly, or monthly) are currently not supported in this version, as the model needs a minimum context length of 512 or 1024.
3. Enabling any upsampling or prepending zeros to virtually increase the context length for shorter-length datasets is not recommended and will
   impact the model performance.

---

## Factors
This section addresses variables that may impact the model's performance.

### Relevant factors:
Relevant factors include context length, forecast length, and time-series resolution (minutely to hourly). Data scaling is also a crucial factor, requiring users to externally standardize their data.

### Evaluation factors:
Not available.

---

## Metrics
This section describes how the model's performance is evaluated.

### Model performance measures:
The model's performance is measured by its ability to outperform popular benchmarks in zero-shot and few-shot forecasting. Specific metrics are implied through comparisons like "outperforms by 7-12%", "outperforms by 24%", etc., suggesting standard forecasting accuracy metrics are used in benchmarking.

### Decision thresholds:
Not available.

### Variation approaches:
Model performance is evaluated through benchmarks, with detailed results available in provided notebooks.

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
The TTM models were trained on a collection of datasets from the Monash Time Series Forecasting repository. The datasets used include:
 - Australian Electricity Demand: https://zenodo.org/records/4659727
 - Australian Weather: https://zenodo.org/records/4654822
 - Bitcoin dataset: https://zenodo.org/records/5122101
 - KDD Cup 2018 dataset: https://zenodo.org/records/4656756
 - London Smart Meters: https://zenodo.org/records/4656091
 - Saugeen River Flow: https://zenodo.org/records/4656058
 - Solar Power: https://zenodo.org/records/4656027
 - Sunspots: https://zenodo.org/records/4654722
 - Solar: https://zenodo.org/records/4656144
 - US Births: https://zenodo.org/records/4656049
 - Wind Farms Production data: https://zenodo.org/records/4654858
 - Wind Power: https://zenodo.org/records/4656032

### Motivation:
The datasets are publicly available time series data.

### Preprocessing:
Data scaling was a part of the training data preprocessing.

---

## Quantitative Analyses
This section presents disaggregated evaluation results.

### Unitary results:
Benchmark Highlights:
- TTM (with less than 1 Million parameters) outperforms the following popular Pre-trained SOTAs demanding several hundred Million to Billions of parameters [paper](https://arxiv.org/pdf/2401.03955v5.pdf):
  - *GPT4TS (NeurIPS 23) by 7-12% in few-shot forecasting*
  - *LLMTime (NeurIPS 23) by 24% in zero-shot forecasting*.
  - *SimMTM (NeurIPS 23) by 17% in few-shot forecasting*.
  - *Time-LLM (ICLR 24) by 2-8% in few-shot forecasting*
  - *UniTime (WWW 24) by 27% in zero-shot forecasting.*
- Zero-shot results of TTM surpass the *few-shot results of many popular SOTA approaches* including
  PatchTST (ICLR 23), PatchTSMixer (KDD 23), TimesNet (ICLR 23), DLinear (AAAI 23) and FEDFormer (ICML 22).
- TTM (1024-96, released in this model card with 1M parameters) outperforms pre-trained MOIRAI-Small (14M parameters) by 10%, MOIRAI-Base (91M parameters) by 2% and
  MOIRAI-Large (311M parameters) by 3% on zero-shot forecasting (horizon = 96). [[notebook]](https://github.com/IBM/tsfm/blob/main/notebooks/hfdemo/tinytimemixer/ttm_benchmarking_1024_96.ipynb)
- TTM quick fine-tuning also outperforms the competitive statistical baselines (Statistical ensemble and S-Naive) in
  M4-hourly dataset which existing pretrained TS models are finding difficult to outperform. [[notebook]](https://github.com/IBM/tsfm/blob/main/notebooks/hfdemo/tinytimemixer/ttm_m4_hourly.ipynb)

### Intersectional results:
Not available.

---

## Memory or Hardware Requirements
This section outlines the memory or hardware requirements for loading, deploying, and training the model.

### Loading Requirements:
Zeroshot, fine-tuning, and inference tasks using TTM can easily be executed in 1 GPU machine or laptops.

### Deploying Requirements:
Zeroshot, fine-tuning, and inference tasks using TTM can easily be executed in 1 GPU machine or laptops.

### Training or Fine-tuning Requirements:
Finetuning and inference tasks can be done on a 1 GPU machine. Pretraining TTMs takes 3-6 hours using 6 A100 GPUs.

---

## Ethical Considerations
This section discusses the ethical considerations in model development, including challenges, risks, and solutions.

Not available.

---

## Caveats and Recommendations
This section lists unresolved issues and provides guidance for users.

### Caveats:
- The current open-source version supports only minutely and hourly resolutions.
- Other lower resolutions (weekly, or monthly) are currently not supported.
- Enabling upsampling or prepending zeros to virtually increase the context length for shorter-length datasets is not recommended.
- Script for Finetuning with cross-channel correlation support - to be added soon

### Recommendations:
- Users have to externally standard scale their data independently for every channel before feeding it to the model.
- Refer to [TSP](https://github.com/IBM/tsfm/blob/main/tsfm_public/toolkit/time_series_preprocessor.py) for data scaling utility.
- Use the model for minutely and hourly resolutions.

---

## Additional Information

<p align="center" width="100%">
<img src="ttm_image.webp" width="600">
</p>

TinyTimeMixers (TTMs) are compact pre-trained models for Multivariate Time-Series Forecasting, open-sourced by IBM Research.
**With less than 1 Million parameters, TTM introduces the notion of the first-ever “tiny” pre-trained models for Time-Series Forecasting.**

TTM outperforms several popular benchmarks demanding billions of parameters in zero-shot and few-shot forecasting. TTMs are lightweight
forecasters, pre-trained on publicly available time series data with various augmentations. TTM provides state-of-the-art zero-shot forecasts and can easily be
fine-tuned for multi-variate forecasts with just 5% of the training data to be competitive. Refer to our [paper](https://arxiv.org/pdf/2401.03955v5.pdf) for more details.

**The current open-source version supports point forecasting use-cases specifically ranging from minutely to hourly resolutions
(Ex. 10 min, 15 min, 1 hour.).**

**Note that zeroshot, fine-tuning and inference tasks using TTM can easily be executed in 1 GPU machine or in laptops too!!**

**Recent updates:** We have developed more sophisticated variants of TTMs (TTM-B, TTM-E and TTM-A), featuring extended benchmarks that compare them with some of the latest models
such as TimesFM, Moirai, Chronos, Lag-llama, and Moment. For full details, please refer to the latest version of our [paper](https://arxiv.org/pdf/2401.03955.pdf).
Stay tuned for the release of the model weights for these newer variants.

TTM takes only a *few seconds for zeroshot/inference* and a *few minutes for finetuning* in 1 GPU machine, as
  opposed to long timing-requirements and heavy computing infra needs of other existing pre-trained models.

Hence, in this model card, we plan to release several pre-trained
TTMs that can cater to many common forecasting settings in practice. Additionally, we have released our source code along with
our pretraining scripts that users can utilize to pretrain models on their own.

Each pre-trained model will be released in a different branch name in this model card. Kindly access the required model using our
getting started [notebook](https://github.com/IBM/tsfm/blob/main/notebooks/hfdemo/ttm_getting_started.ipynb) mentioning the branch name.

- **512-96:** Given the last 512 time-points (i.e. context length), this model can forecast up to next 96 time-points (i.e. forecast length)
  in future. This model is targeted towards a forecasting setting of context length 512 and forecast length 96 and
  recommended for hourly and minutely resolutions (Ex. 10 min, 15 min, 1 hour, etc).   (branch name: main)

- **1024-96:** Given the last 1024 time-points (i.e. context length), this model can forecast up to next 96 time-points (i.e. forecast length)
  in future. This model is targeted towards a long forecasting setting of context length 1024 and forecast length 96 and
  recommended for hourly and minutely resolutions (Ex. 10 min, 15 min, 1 hour, etc). (branch name: 1024-96-v1)

- Stay tuned for more models !

IBM Public Repository Disclosure:

All content in this repository including code has been provided by IBM under the associated
open source software license and IBM is under no obligation to provide enhancements,
updates, or support. IBM developers produced this code as an
open source project (not as an IBM product), and IBM makes no assertions as to
the level of quality nor security, and will not be maintaining this code going forward.