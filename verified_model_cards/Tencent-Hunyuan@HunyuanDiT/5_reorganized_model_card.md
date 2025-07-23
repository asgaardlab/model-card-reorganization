## Model Details
This section provides fundamental information about the model, helping stakeholders understand its context and key characteristics.

### Person or organization developing model:
Tencent Hunyuan, Chinese University of Hong Kong, Shenzhen Campus of Sun Yat-sen University.
This model is developed by researchers from Tencent Hunyuan and academic institutions. Tencent Hunyuan is known for its innovative AI products. More details can be found on the [project page](https://dit.hunyuan.tencent.com/) and the [Hugging Face repository](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT).

### Model date:
The model development is ongoing, with key milestones around May 2024, indicated by the publication date of the [Hunyuan-DiT paper](https://arxiv.org/abs/2405.08748). The [DialogGen paper](https://arxiv.org/abs/2403.08857) was published in March 2024. Some features like Distillation Version, TensorRT Version, and Training code are marked as "Coming soon" or "Coming later".

### Model version:
This is the initial release of Hunyuan-DiT, a powerful multi-resolution diffusion transformer with fine-grained Chinese understanding.

### Model type:
Hunyuan-DiT is a text-to-image diffusion model in the latent space, parameterized with a transformer architecture (Diffusion Transformer - DiT). It is a Chinese-English Bilingual DiT Architecture, leveraging a combination of pre-trained bilingual (English and Chinese) CLIP and multilingual T5 encoder for text prompts. It also includes DialogGen, a prompt enhancement model. The model size is around 1.5B parameters for Hunyuan-DiT, 7.0B for DialogGen, 1.6B for mT5, and 350M for CLIP. Context length details are not explicitly provided.

### Training details:
Hunyuan-DiT is trained using a carefully designed data pipeline and a Multimodal Large Language Model (MLLM) to refine image captions for fine-grained language understanding. The training process involves iterative model optimization and holistic human evaluation protocol with more than 50 professional human evaluators.

### Paper or other resource for more information:
- [Hunyuan-DiT Paper](https://arxiv.org/abs/2405.08748)
- [DialogGen Paper](https://arxiv.org/abs/2403.08857)
- [Project Page](https://dit.hunyuan.tencent.com/)
- [Hugging Face Repository](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT)
- [DialogGen GitHub Repository](https://github.com/Centaurusalpha/DialogGen)

These resources provide in-depth information about the model architecture, methodology, and performance.

### Citation details:
```BibTeX
@misc{li2024hunyuandit,
      title={Hunyuan-DiT: A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding},
      author={Zhimin Li and Jianwei Zhang and Qin Lin and Jiangfeng Xiong and Yanxin Long and Xinchi Deng and Yingfang Zhang and Xingchao Liu and Minbin Huang and Zedong Xiao and Dayou Chen and Jiajun He and Jiahao Li and Wenyue Li and Chen Zhang and Rongwei Quan and Jianxiang Lu and Jiabin Huang and Xiaoyan Yuan and Xiaoxiao Zheng and Yixuan Li and Jihong Zhang and Chao Zhang and Meng Chen and Jie Liu and Zheng Fang and Weiyan Wang and Jinbao Xue and Yangyu Tao and Jianchen Zhu and Kai Liu and Sihuan Lin and Yifu Sun and Yun Li and Dongdong Wang and Mingtao Chen and Zhichao Hu and Xiao Xiao and Yan Chen and Yuhong Liu and Wei Liu and Di Wang and Yong Yang and Jie Jiang and Qinglin Lu},
      year={2024},
      eprint={2405.08748},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{huang2024dialoggen,
  title={DialogGen: Multi-modal Interactive Dialogue System for Multi-turn Text-to-Image Generation},
  author={Huang, Minbin and Long, Yanxin and Deng, Xinchi and Chu, Ruihang and Xiong, Jiangfeng and Liang, Xiaodan and Cheng, Hong and Lu, Qinglin and Liu, Wei},
  journal={arXiv preprint arXiv:2403.08857},
  year={2024}
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
Hunyuan-DiT is primarily intended for text-to-image generation, with a focus on fine-grained understanding of both English and Chinese. It is capable of multi-turn text-to-image generation, allowing for interactive refinement of images through dialogue. The model also includes DialogGen for prompt enhancement, improving the quality of generated images. Example use cases include generating images from simple or complex text prompts, creating cyberpunk-style paintings of sports cars, or drawing a pig in a suit. The input is text prompts in English or Chinese, and the output is generated images.

### Primary intended users:
The primary intended users are researchers and developers in the field of AI and computer vision, as well as general users interested in text-to-image generation technology. The [Tencent Hunyuan Bot](https://hunyuan.tencent.com/bot/chat) provides a platform for general users to explore the model's capabilities.

### Out-of-scope uses:
The model is specifically designed for text-to-image generation. Uses outside of this domain, such as tasks unrelated to image synthesis from textual descriptions, are considered out-of-scope. Misuse could include generating inappropriate or harmful content, which is a general risk with text-to-image models.

---

## How to Use
This section outlines how to use the model.

To use Hunyuan-DiT, you can utilize either Gradio interface or command-line scripts.

**Using Gradio:**
Run `python app/hydit_app.py` to start a Chinese UI by default. Use flags like `--infer-mode fa` for Flash Attention acceleration, `--no-enhance` to disable prompt enhancement, or `--lang en` for English UI.

**Using Command Line:**
Run `sample_t2i.py` with prompts. Examples:
- `python sample_t2i.py --prompt "渔舟唱晚"` (Prompt Enhancement + Text-to-Image, Torch mode)
- `python sample_t2i.py --prompt "渔舟唱晚" --no-enhance` (Only Text-to-Image, Torch mode)
- `python sample_t2i.py --infer-mode fa --prompt "渔舟唱晚"` (Only Text-to-Image, Flash Attention mode)
- `python sample_t2i.py --prompt "渔舟唱晚" --image-size 1280 768` (Generate with specific image size)

**Configurations:**
Key arguments include:
- `--prompt`: Text prompt for image generation.
- `--image-size`: Size of the generated image (e.g., 1024 1024).
- `--seed`: Random seed (default: 42).
- `--infer-steps`: Number of sampling steps (default: 100).
- `--negative`: Negative prompt.
- `--infer-mode`: Inference mode (`torch` or `fa`).
- `--sampler`: Diffusion sampler (`ddpm`, `ddim`, or `dpmms`).
- `--no-enhance`: Disable prompt enhancement model.
- `--model-root`: Model checkpoints directory (default: `ckpts`).
- `--load-key`: Model loading key (`ema` or `module`).

Refer to [example_prompts.txt](example_prompts.txt) for more prompt examples and the [Hugging Face repository](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT) for model files and further instructions.

---

## Factors
This section addresses variables that may impact the model's performance.

### Relevant factors:
The model's performance is influenced by its ability to understand both Chinese and English languages, handle long text inputs, and engage in multi-turn text-to-image generation. These factors are crucial for its intended use in diverse linguistic contexts and interactive applications.

### Evaluation factors:
The evaluation factors analyzed are Text-Image Consistency, Excluding AI Artifacts, Subject Clarity, and Aesthetic quality. These factors were selected to comprehensively assess the visual fidelity, artifact reduction, subject depiction, and overall aesthetic appeal of the generated images.

---

## Metrics
This section describes how the model's performance is evaluated.

### Model performance measures:
The model's performance is measured using the following metrics, evaluated by human evaluators:
- **Text-Image Consistency (%)**: Measures how well the generated image aligns with the text prompt.
- **Excluding AI Artifacts (%)**: Measures the absence of unwanted artifacts in the generated image.
- **Subject Clarity (%)**: Measures the clarity and recognizability of the main subject in the image.
- **Aesthetics (%)**: Measures the overall aesthetic appeal and visual quality of the image.
- **Overall (%)**: A combined overall score reflecting the general quality of the generated images.
These metrics are suitable for evaluating text-to-image models as they directly assess the key aspects of image generation quality from a human perceptual perspective.

### Decision thresholds:
Not available.

### Variation approaches:
Performance metrics were obtained through a holistic human evaluation protocol involving more than 50 professional human evaluators. This approach ensures a robust and reliable assessment of the model's generation capabilities by averaging evaluations from multiple experts.

---

## Evaluation Data
This section provides details about the datasets used to evaluate the model.

### Datasets:
A 4-dimensional test set was constructed to evaluate the model's generation capabilities. The specifics of this dataset (size, diversity, source, public availability or proprietary) are not detailed in the provided content.

### Motivation:
The dataset was chosen to comprehensively compare Hunyuan-DiT's generation capabilities against other models, including SDXL, PixArt-α, Playground 2.5, SD 3, MidJourney v6, and DALL-E 3.

### Preprocessing:
Not available.

---

## Training Data
This section provides details about the datasets used to train the model.

### Datasets:
The model was trained using a data pipeline, and captions in the training data were refined using a Multimodal Large Language Model (MLLM) to enhance fine-grained language understanding. Further details about the dataset's size, structure, features, diversity, and public availability are not provided.

### Motivation:
The datasets were chosen to enable the model to achieve fine-grained language understanding and for iterative model optimization. The use of MLLM-refined captions aims to improve the model's ability to generate images that accurately reflect the nuances of text prompts.

### Preprocessing:
Preprocessing included refining captions of images using a Multimodal Large Language Model (MLLM). This step was crucial for improving the model's ability to understand and generate images based on complex textual descriptions.

---

## Quantitative Analyses
This section presents disaggregated evaluation results.

### Unitary results:
| Model            | Open Source | Text-Image Consistency (%) | Excluding AI Artifacts (%) | Subject Clarity (%) | Aesthetics (%) | Overall (%) |
|------------------|-------------|--------------------------|----------------------------|---------------------|----------------|-------------|
| SDXL             | ✔           | 64.3                     | 60.6                       | 91.1                | 76.3           | 42.7        |
| PixArt-α         | ✔           | 68.3                     | 60.9                       | 93.2                | 77.5           | 45.5        |
| Playground 2.5   | ✔           | 71.9                     | 70.8                       | 94.9                | 83.3           | 54.3        |
| SD 3             | ✘           | 77.1                     | 69.3                       | 94.6                | 82.5           | 56.7        |
| MidJourney v6    | ✘           | 73.5                     | 80.2                       | 93.5                | 87.2           | 63.3        |
| DALL-E 3         | ✘           | 83.9                     | 80.3                       | 96.5                | 89.4           | 71.0        |
| **Hunyuan-DiT**  | ✔           | 74.2                     | 74.3                       | 95.4                | 86.6           | 59.0        |

### Intersectional results:
Not available.

---

## Memory or Hardware Requirements
This section outlines the memory or hardware requirements for loading, deploying, and training the model.

### Loading Requirements:
Loading requirements are not explicitly specified but are implied by the deployment requirements.

### Deploying Requirements:
- **GPU Memory**: Minimum 11GB, Recommended 32GB for better generation quality.
- **GPU**: NVIDIA V100 or A100 with CUDA support.
- **Operating System**: Linux (tested).

### Training or Fine-tuning Requirements:
Training or fine-tuning would likely require at least 32GB GPU memory, as recommended for better generation quality, and a compatible NVIDIA GPU (V100/A100).

---

## Ethical Considerations
This section discusses the ethical considerations in model development, including challenges, risks, and solutions.

Sensitive data usage, potential risks, risk mitigation strategies, and fraught use cases are not explicitly discussed in the provided model card content. However, the focus on human evaluation and model comparisons suggests an awareness of the need for responsible development and evaluation.

---

## Caveats and Recommendations
This section lists unresolved issues and provides guidance for users.

### Caveats:
- Distillation Version: Coming soon.
- TensorRT Version: Coming soon.
- Training code: Coming later.
- Evaluation datasets details are not fully specified.

### Recommendations:
- For optimal performance and better generation quality, it is recommended to use a GPU with at least 32GB of memory.
- Explore the [Tencent Hunyuan Bot](https://hunyuan.tencent.com/bot/chat) to experience the model's capabilities and provide feedback.
- Stay tuned for upcoming releases of Distillation Version, TensorRT Version, and Training code for enhanced performance and broader accessibility.

---

## Additional Information
<!-- ## **HunyuanDiT** -->

<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent/HunyuanDiT/main/asset/logo.png"  height=100>
</p>

# Hunyuan-DiT : A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding

This repo contains PyTorch model definitions, pre-trained weights and inference/sampling code for our paper exploring Hunyuan-DiT. You can find more visualizations on our [project page](https://dit.hunyuan.tencent.com/).

## 🔥🔥🔥 Tencent Hunyuan Bot

Welcome to [Tencent Hunyuan Bot](https://hunyuan.tencent.com/bot/chat), where you can explore our innovative products! Just input the suggested prompts below or any other **imaginative prompts containing drawing-related keywords** to activate the Hunyuan text-to-image generation feature.  You can use **simple prompts** as well as **multi-turn language interactions** to create the picture. Unleash your creativity and create any picture you desire, **all for free!**
> 画一只穿着西装的猪
>
> draw a pig in a suit
>
> 生成一幅画，赛博朋克风，跑车
>
> generate a painting, cyberpunk style, sports car

## 📑 Open-source Plan

- Hunyuan-DiT (Text-to-Image Model)
  - [x] Inference
  - [x] Checkpoints
  - [ ] Distillation Version (Coming soon ⏩️)
  - [ ] TensorRT Version (Coming soon ⏩️)
  - [ ] Training (Coming later ⏩️)
- [DialogGen](https://github.com/Centaurusalpha/DialogGen) (Prompt Enhancement Model)
  - [x] Inference
- [X] Web Demo (Gradio)
- [X] Cli Demo

## Contents
- [Hunyuan-DiT](#hunyuan-dit--a-powerful-multi-resolution-diffusion-transformer-with-fine-grained-chinese-understanding)
  - [Abstract](#abstract)
  - [🎉 Hunyuan-DiT Key Features](#-hunyuan-dit-key-features)
    - [Chinese-English Bilingual DiT Architecture](#chinese-english-bilingual-dit-architecture)
    - [Multi-turn Text2Image Generation](#multi-turn-text2image-generation)
  - [📈 Comparisons](#-comparisons)
  - [🎥 Visualization](#-visualization)
  - [📜 Requirements](#-requirements)
  - [🛠 Dependencies and Installation](#%EF%B8%8F-dependencies-and-installation)
  - [🧱 Download Pretrained Models](#-download-pretrained-models)
  - [🔑 Inference](#-inference)
    - [Using Gradio](#using-gradio)
    - [Using Command Line](#using-command-line)
    - [More Configurations](#more-configurations)
  - [🔗 BibTeX](#-bibtex)

## **Abstract**

We present Hunyuan-DiT, a text-to-image diffusion transformer with fine-grained understanding of both English and Chinese. To construct Hunyuan-DiT, we carefully designed the transformer structure, text encoder, and positional encoding. We also build from scratch a whole data pipeline to update and evaluate data for iterative model optimization. For fine-grained language understanding, we train a Multimodal Large Language Model to refine the captions of the images. Finally, Hunyuan-DiT can perform multi-round multi-modal dialogue with users, generating and refining images according to the context.
Through our carefully designed holistic human evaluation protocol with more than 50 professional human evaluators, Hunyuan-DiT sets a new state-of-the-art in Chinese-to-image generation compared with other open-source models.


## 🎉 **Hunyuan-DiT Key Features**
### **Chinese-English Bilingual DiT Architecture**
Hunyuan-DiT is a diffusion model in the latent space, as depicted in figure below. Following the Latent Diffusion Model, we use a pre-trained Variational Autoencoder (VAE) to compress the images into low-dimensional latent spaces and train a diffusion model to learn the data distribution with diffusion models. Our diffusion model is parameterized with a transformer. To encode the text prompts, we leverage a combination of pre-trained bilingual (English and Chinese) CLIP and multilingual T5 encoder.
<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent/HunyuanDiT/main/asset/framework.png"  height=450>
</p>

### Multi-turn Text2Image Generation
Understanding natural language instructions and performing multi-turn interaction with users are important for a
text-to-image system. It can help build a dynamic and iterative creation process that bring the user’s idea into reality
step by step. In this section, we will detail how we empower Hunyuan-DiT with the ability to perform multi-round
conversations and image generation. We train MLLM to understand the multi-round user dialogue
and output the new text prompt for image generation.
<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent/HunyuanDiT/main/asset/mllm.png"  height=300>
</p>

## 📈 Comparisons
In order to comprehensively compare the generation capabilities of HunyuanDiT and other models, we constructed a 4-dimensional test set, including Text-Image Consistency, Excluding AI Artifacts, Subject Clarity, Aesthetic. More than 50 professional evaluators performs the evaluation.


## 🎥 Visualization

* **Chinese Elements**
<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent/HunyuanDiT/main/asset/chinese elements understanding.png"  height=220>
</p>

* **Long Text Input**


<p align="center">
  <img src="https://raw.githubusercontent.com/Tencent/HunyuanDiT/main/asset/long text understanding.png"  height=310>
</p>

* **Multi-turn Text2Image Generation**

[demo video](https://youtu.be/4AaHrYnuIcE)

---

## 📜 Requirements

This repo consists of DialogGen (a prompt enhancement model) and Hunyuan-DiT (a text-to-image model).

The following table shows the requirements for running the models (The TensorRT version will be updated soon):

|          Model           | TensorRT | Batch Size | GPU Memory |    GPU    |
|:------------------------:|:--------:|:----------:|:----------:|:---------:|
| DialogGen + Hunyuan-DiT  |    ✘     |     1      |    32G     | V100/A100 |
|       Hunyuan-DiT        |    ✘     |     1      |    11G     | V100/A100 |

<!-- | DialogGen + Hunyuan-DiT  |    ✔     |     1      |     ?      |   A100    |
|       Hunyuan-DiT        |    ✔     |     1      |     ?      |   A100    | -->

* An NVIDIA GPU with CUDA support is required.
  * We have tested V100 and A100 GPUs.
  * **Minimum**: The minimum GPU memory required is 11GB.
  * **Recommended**: We recommend using a GPU with 32GB of memory for better generation quality.
* Tested operating system: Linux

## 🛠️ Dependencies and Installation

Begin by cloning the repository:
```bash
git clone https://github.com/tencent/HunyuanDiT
cd HunyuanDiT
```

We provide an `environment.yml` file for setting up a Conda environment.
Conda's installation instructions are available [here](https://docs.anaconda.com/free/miniconda/index.html).

```bash
# 1. Prepare conda environment
conda env create -f environment.yml

# 2. Activate the environment
conda activate HunyuanDiT

# 3. Install pip dependencies
python -m pip install -r requirements.txt

# 4. (Optional) Install flash attention v2 for acceleration (requires CUDA 11.6 or above)
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.1.2.post3
```

## 🧱 Download Pretrained Models
To download the model, first install the huggingface-cli. (Detailed instructions are available [here](https://huggingface.co/docs/huggingface_hub/guides/cli).)

```bash
python -m pip install "huggingface_hub[cli]"
```

Then download the model using the following commands:

```bash
# Create a directory named 'ckpts' where the model will be saved, fulfilling the prerequisites for running the demo.
mkdir ckpts
# Use the huggingface-cli tool to download the model.
# The download time may vary from 10 minutes to 1 hour depending on network conditions.
huggingface-cli download Tencent-Hunyuan/HunyuanDiT --local-dir ./ckpts
```
Note：If an `No such file or directory: 'ckpts/.huggingface/.gitignore.lock'` like error occurs during the download process, you can ignore the error and retry the command by executing `huggingface-cli download Tencent-Hunyuan/HunyuanDiT --local-dir ./ckpts`

All models will be automatically downloaded. For more information about the model, visit the Hugging Face repository [here](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT).

|       Model        | #Params |                                              Download URL                                               |
|:------------------:|:-------:|:-------------------------------------------------------------------------------------------------------:|
|        mT5         |  1.6B   |               [mT5](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/mt5)                |
|        CLIP        |  350M   |        [CLIP](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/clip_text_encoder)        |
|     DialogGen      |  7.0B   |           [DialogGen](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/dialoggen)            |
| sdxl-vae-fp16-fix  |   83M   | [sdxl-vae-fp16-fix](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/sdxl-vae-fp16-fix)  |
|    Hunyuan-DiT     |  1.5B   |          [Hunyuan-DiT](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/model)           |


---
