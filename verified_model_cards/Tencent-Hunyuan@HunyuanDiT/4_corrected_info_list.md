## Hunyuan-DiT
- ![HunyuanDiT Logo](https://raw.githubusercontent.com/Tencent/HunyuanDiT/main/asset/logo.png)
- Hunyuan-DiT is a powerful multi-resolution diffusion transformer.
- Hunyuan-DiT has fine-grained Chinese understanding.
- This repository contains PyTorch model definitions.
- The repository includes pre-trained weights.
- The repository includes inference/sampling code for Hunyuan-DiT.
- More visualizations can be found on the [project page](https://dit.hunyuan.tencent.com/).
- The paper titled [**Hunyuan-DiT : A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding**](https://arxiv.org/abs/2405.08748) is authored by Zhimin Li, Jianwei Zhang, Qin Lin, Jiangfeng Xiong, Yanxin Long, Xinchi Deng, Yingfang Zhang, Xingchao Liu, Minbin Huang, Zedong Xiao, Dayou Chen, Jiajun He, Jiahao Li, Wenyue Li, Chen Zhang, Rongwei Quan, Jianxiang Lu, Jiabin Huang, Xiaoyan Yuan, Xiaoxiao Zheng, Yixuan Li, Jihong Zhang, Chao Zhang, Meng Chen, Jie Liu, Zheng Fang, Weiyan Wang, Jinbao Xue, Yangyu Tao, JianChen Zhu, Kai Liu, Sihuan Lin, Yifu Sun, Yun Li, Dongdong Wang, Zhichao Hu, Xiao Xiao, Yan Chen, Yuhong Liu, Wei Liu, Di Wang, Yong Yang, Jie Jiang, Qinglin Lu.
- The paper is associated with Tencent Hunyuan.
- Another paper titled [**DialogGen: Multi-modal Interactive Dialogue System for Multi-turn Text-to-Image Generation**](https://arxiv.org/abs/2403.08857) is authored by Minbin Huang, Yanxin Long, Xinchi Deng, Ruihang Chu, Jiangfeng Xiong, Xiaodan Liang, Hong Cheng, Qinglin Lu, Wei Liu.
- The second paper is associated with Chinese University of Hong Kong, Tencent Hunyuan, Shenzhen Campus of Sun Yat-sen University.

## 🔥🔥🔥 Tencent Hunyuan Bot
- Welcome to [Tencent Hunyuan Bot](https://hunyuan.tencent.com/bot/chat).
- Users can explore innovative products through the bot.
- Users can input suggested prompts or any imaginative prompts containing drawing-related keywords.
- The prompt activates the Hunyuan text-to-image generation feature.
- Users can use simple prompts as well as multi-turn language interactions to create pictures.
- Users can create any picture they desire, all for free.
- Example prompts include:
  - 画一只穿着西装的猪 (draw a pig in a suit)
  - 生成一幅画，赛博朋克风，跑车 (generate a painting, cyberpunk style, sports car)

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
- [X] CLI Demo 

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
- Hunyuan-DiT is a text-to-image diffusion transformer.
- Hunyuan-DiT has fine-grained understanding of both English and Chinese.
- The transformer structure, text encoder, and positional encoding were carefully designed.
- A whole data pipeline to update and evaluate data was built from scratch for iterative model optimization.
- A Multimodal Large Language Model was trained to refine the captions of the images for fine-grained language understanding.
- Hunyuan-DiT can perform multi-round multi-modal dialogue with users.
- Hunyuan-DiT generates and refines images according to the context.
- Hunyuan-DiT sets a new state-of-the-art in Chinese-to-image generation.
- The evaluation was conducted with more than 50 professional human evaluators.

## 🎉 **Hunyuan-DiT Key Features**
### **Chinese-English Bilingual DiT Architecture**
- Hunyuan-DiT is a diffusion model in the latent space.
- A pre-trained Variational Autoencoder (VAE) is used to compress images into low-dimensional latent spaces.
- A diffusion model is trained to learn the data distribution.
- The diffusion model is parameterized with a transformer.
- A combination of pre-trained bilingual (English and Chinese) CLIP and multilingual T5 encoder is used to encode text prompts.
- ![Hunyuan-DiT Architecture](https://raw.githubusercontent.com/Tencent/HunyuanDiT/main/asset/framework.png)

### Multi-turn Text2Image Generation
- Understanding natural language instructions and performing multi-turn interaction with users are important for a text-to-image system.
- These helps build a dynamic and iterative creation process.
- Hunyuan-DiT is empowered to perform multi-round conversations and image generation.
- MLLM is trained to understand multi-round user dialogue.
- MLLM outputs new text prompts for image generation.
- ![Multi-turn Text2Image Generation](https://raw.githubusercontent.com/Tencent/HunyuanDiT/main/asset/mllm.png)

## 📈 Comparisons
- A 4-dimensional test set was constructed to compare the generation capabilities of Hunyuan-DiT and other models.
- The test set includes Text-Image Consistency, Excluding AI Artifacts, Subject Clarity, and Aesthetic.
- More than 50 professional evaluators performed the evaluation.
- Table comparing models:
  <p align="center">
  <table> 
  <thead> 
  <tr> 
      <th rowspan="2">Model</th> <th rowspan="2">Open Source</th> <th>Text-Image Consistency (%)</th> <th>Excluding AI Artifacts (%)</th> <th>Subject Clarity (%)</th> <th rowspan="2">Aesthetics (%)</th> <th rowspan="2">Overall (%)</th> 
  </tr> 
  </thead> 
  <tbody> 
  <tr> 
      <td>SDXL</td> <td> ✔ </td> <td>64.3</td> <td>60.6</td> <td>91.1</td> <td>76.3</td> <td>42.7</td> 
  </tr> 
  <tr> 
      <td>PixArt-α</td> <td> ✔</td> <td>68.3</td> <td>60.9</td> <td>93.2</td> <td>77.5</td> <td>45.5</td> 
  </tr> 
  <tr> 
      <td>Playground 2.5</td> <td>✔</td> <td>71.9</td> <td>70.8</td> <td>94.9</td> <td>83.3</td> <td>54.3</td> 
  </tr> 
  <tr> 
      <td>SD 3</td> <td>&#10008</td> <td>77.1</td> <td>69.3</td> <td>94.6</td> <td>82.5</td> <td>56.7</td> 
  </tr> 
  <tr> 
      <td>MidJourney v6</td><td>&#10008</td> <td>73.5</td> <td>80.2</td> <td>93.5</td> <td>87.2</td> <td>63.3</td> 
  </tr> 
  <tr> 
      <td>DALL-E 3</td><td>&#10008</td> <td>83.9</td> <td>80.3</td> <td>96.5</td> <td>89.4</td> <td>71.0</td> 
  </tr> 
  <tr style="font-weight: bold; background-color: #f2f2f2;"> 
      <td>Hunyuan-DiT</td><td>✔</td> <td>74.2</td> <td>74.3</td> <td>95.4</td> <td>86.6</td> <td>59.0</td> 
  </tr>
  </tbody>
  </table>
  </p>

## 🎥 Visualization
- **Chinese Elements**
  - ![Chinese Elements Understanding](https://raw.githubusercontent.com/Tencent/HunyuanDiT/main/asset/chinese elements understanding.png)
- **Long Text Input**
  - ![Long Text Understanding](https://raw.githubusercontent.com/Tencent/HunyuanDiT/main/asset/long text understanding.png)
- **Multi-turn Text2Image Generation**
  - [Demo video](https://youtu.be/4AaHrYnuIcE)

## 📜 Requirements
- The repository consists of DialogGen (a prompt enhancement model) and Hunyuan-DiT (a text-to-image model).
- The following table shows the requirements for running the models:
  |          Model           | TensorRT | Batch Size | GPU Memory |    GPU    |
  |:------------------------:|:--------:|:----------:|:----------:|:---------:|
  | DialogGen + Hunyuan-DiT  |    ✘     |     1      |    32G     | V100/A100 |
  |       Hunyuan-DiT        |    ✘     |     1      |    11G     | V100/A100 |
- An NVIDIA GPU with CUDA support is required.
- Tested GPUs include V100 and A100.
- Minimum GPU memory required is 11GB.
- Recommended GPU memory is 32GB for better generation quality.
- Tested operating system is Linux.

## 🛠️ Dependencies and Installation
- Begin by cloning the repository:
  ```bash
  git clone https://github.com/tencent/HunyuanDiT
  cd HunyuanDiT
  ```
- An `environment.yml` file is provided for setting up a Conda environment.
- Conda's installation instructions are available [here](https://docs.anaconda.com/free/miniconda/index.html).
- Steps to set up the environment:
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
- To download the model, first install the huggingface-cli. Detailed instructions are available [here](https://huggingface.co/docs/huggingface_hub/guides/cli).
  ```bash
  python -m pip install "huggingface_hub[cli]"
  ```
- Download the model using the following commands:
  ```bash
  # Create a directory named 'ckpts' where the model will be saved.
  mkdir ckpts
  # Use the huggingface-cli tool to download the model.
  # The download time may vary from 10 minutes to 1 hour depending on network conditions.
  huggingface-cli download Tencent-Hunyuan/HunyuanDiT --local-dir ./ckpts
  ```
- Note: If a `No such file or directory: 'ckpts/.huggingface/.gitignore.lock'` like error occurs during the download process, you can ignore the error
- Retry the command by executing `huggingface-cli download Tencent-Hunyuan/HunyuanDiT --local-dir ./ckpts`.
- All models will be automatically downloaded.
- For more information about the model, visit the Hugging Face repository [here](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT).
- Table of models and download URLs:
  |       Model        | #Params |                                              Download URL                                               |
  |:------------------:|:-------:|:-------------------------------------------------------------------------------------------------------:|
  |        mT5         |  1.6B   |               [mT5](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/mt5)                |
  |        CLIP        |  350M   |        [CLIP](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/clip_text_encoder)        |
  |     DialogGen      |  7.0B   |           [DialogGen](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/dialoggen)            |
  | sdxl-vae-fp16-fix  |   83M   | [sdxl-vae-fp16-fix](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/sdxl-vae-fp16-fix)  |
  |    Hunyuan-DiT     |  1.5B   |          [Hunyuan-DiT](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/tree/main/t2i/model)           |

## 🔑 Inference
### Using Gradio
- Ensure the conda environment is activated before running the following command.
  ```shell
  # By default, we start a Chinese UI.
  python app/hydit_app.py

  # Using Flash Attention for acceleration.
  python app/hydit_app.py --infer-mode fa

  # Disable the enhancement model if the GPU memory is insufficient.
  # The enhancement will be unavailable until you restart the app without the `--no-enhance` flag.
  python app/hydit_app.py --no-enhance

  # Start with English UI
  python app/hydit_app.py --lang en
  ```

### Using Command Line
- Three modes to quick start:
  ```bash
  # Prompt Enhancement + Text-to-Image. Torch mode
  python sample_t2i.py --prompt "渔舟唱晚"

  # Only Text-to-Image. Torch mode
  python sample_t2i.py --prompt "渔舟唱晚" --no-enhance

  # Only Text-to-Image. Flash Attention mode
  python sample_t2i.py --infer-mode fa --prompt "渔舟唱晚"

  # Generate an image with other image sizes.
  python sample_t2i.py --prompt "渔舟唱晚" --image-size 1280 768
  ```
- More example prompts can be found in [example_prompts.txt](example_prompts.txt).

### More Configurations
- Useful configurations for easy usage:
  |    Argument     |  Default  |                     Description                     |
  |:---------------:|:---------:|:---------------------------------------------------:|
  |   `--prompt`    |   None    |        The text prompt for image generation         |
  | `--image-size`  | 1024 1024 |           The size of the generated image           |
  |    `--seed`     |    42     |        The random seed for generating images        |
  | `--infer-steps` |    100    |          The number of steps for sampling           |
  |  `--negative`   |     -     |      The negative prompt for image generation       |
  | `--infer-mode`  |   torch   |          The inference mode (torch or fa)           |
  |   `--sampler`   |   ddpm    |    The diffusion sampler (ddpm, ddim, or dpmms)     |
  | `--no-enhance`  |   False   |        Disable the prompt enhancement model         |
  | `--model-root`  |   ckpts   |     The root directory of the model checkpoints     |
  |  `--load-key`   |    ema    | Load the student model or EMA model (ema or module) |

## 🔗 BibTeX
- If you find [Hunyuan-DiT](https://arxiv.org/abs/2405.08748) or [DialogGen](https://arxiv.org/abs/2403.08857) useful for your research and applications, please cite using this BibTeX:
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