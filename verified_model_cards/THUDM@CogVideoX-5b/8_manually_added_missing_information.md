## Model Details
This section provides fundamental information about the model, helping stakeholders understand its context and key characteristics.

### Person or organization developing model:
Not available.

### Model date:
Not available.

### Model version:
CogVideoX-5B (This Repository): Larger model with higher video generation quality and better visual effects.
CogVideoX-2B: Entry-level model, balancing compatibility. Low cost for running and secondary development.

### Model type:
CogVideoX is an open-source version of the video generation model originating from [QingYing](https://chatglm.cn/video?lang=en?fr=osm_cogvideo).
CogVideoX-5B: Larger model with higher video generation quality and better visual effects. Positional Encoding: 3d_rope_pos_embed.
CogVideoX-2B: Positional Encoding: 3d_sincos_pos_embed.

### Training details:
Inference Precision: BF16 (Recommended), FP16, FP32, FP8*, INT8, no support for INT4.
Fine-tuning Precision: BF16.
The 5B model is trained with BF16 precision, and the 2B model is trained with FP16 precision. We recommend using the precision the model was trained with for inference.

### Paper or other resource for more information:
<p align="center">
  <a href="https://huggingface.co/THUDM/CogVideoX-5b/blob/main/README_zh.md">📄 中文阅读</a> |
  <a href="https://huggingface.co/spaces/THUDM/CogVideoX-5B-Space">🤗 Huggingface Space</a> |
  <a href="https://github.com/THUDM/CogVideo">🌐 Github </a> |
  <a href="https://arxiv.org/pdf/2408.06072">📜 arxiv </a>
</p>
Visit our [GitHub](https://github.com/THUDM/CogVideo) for more information.
Welcome to our [github](https://github.com/THUDM/CogVideo), where you will find:
1. More detailed technical details and code explanation.
2. Optimization and conversion of prompt words.
3. Reasoning and fine-tuning of SAT version models, and even pre-release.
4. Project update log dynamics, more interactive opportunities.
5. CogVideoX toolchain to help you better use the model.
6. INT8 model inference code support.

### Citation details:
```
@article{yang2024cogvideox,
  title={CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer},
  author={Yang, Zhuoyi and Teng, Jiayan and Zheng, Wendi and Ding, Ming and Huang, Shiyu and Xu, Jiazheng and Yang, Yuanming and Hong, Wenyi and Zhang, Xiaohan and Feng, Guanyu and others},
  journal={arXiv preprint arXiv:2408.06072},
  year={2024}
}
```

### License:
This model is released under the [CogVideoX LICENSE](LICENSE).

### Contact:
Not available.

---

## Intended Use
This section outlines the intended applications of the model.

### Primary intended uses:
Text-to-Video Diffusion Models.
Prompt Language: English*
Prompt Length Limit: 226 Tokens
Video Length: 6 Seconds
Frame Rate: 8 Frames per Second
Video Resolution: 720 x 480, no support for other resolutions (including fine-tuning)
The model only supports English input; other languages can be translated into English during refinement by a large model.

### Primary intended users:
Not available.

### Out-of-scope uses:
Not available.

---

## How to Use
This section outlines how to use the model.

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Gallery with Captions</title>
    <style>
        .video-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-around;
        }
        .video-item {
            width: 45%;
            margin-bottom: 20px;
            transition: transform 0.3s;
        }
        .video-item:hover {
            transform: scale(1.1);
        }
        .caption {
            text-align: center;
            margin-top: 10px;
            font-size: 11px;
        }
    </style>
</head>
<body>
    <div class="video-container">
        <div class="video-item">
            <video width="100%" controls>
                <source src="https://github.com/user-attachments/assets/cf5953ea-96d3-48fd-9907-c4708752c714" type="video/mp4">
            </video>
            <div class="caption">A garden comes to life as a kaleidoscope of butterflies flutters amidst the blossoms, their delicate wings casting shadows on the petals below. In the background, a grand fountain cascades water with a gentle splendor, its rhythmic sound providing a soothing backdrop. Beneath the cool shade of a mature tree, a solitary wooden chair invites solitude and reflection, its smooth surface worn by the touch of countless visitors seeking a moment of tranquility in nature's embrace.</div>
        </div>
        <div class="video-item">
            <video width="100%" controls>
                <source src="https://github.com/user-attachments/assets/fe0a78e6-b669-4800-8cf0-b5f9b5145b52" type="video/mp4">
            </video>
            <div class="caption">A small boy, head bowed and determination etched on his face, sprints through the torrential downpour as lightning crackles and thunder rumbles in the distance. The relentless rain pounds the ground, creating a chaotic dance of water droplets that mirror the dramatic sky's anger. In the far background, the silhouette of a cozy home beckons, a faint beacon of safety and warmth amidst the fierce weather. The scene is one of perseverance and the unyielding spirit of a child braving the elements.</div>
        </div>
        <div class="video-item">
            <video width="100%" controls>
                <source src="https://github.com/user-attachments/assets/c182f606-8f8c-421d-b414-8487070fcfcb" type="video/mp4">
            </video>
            <div class="caption">A suited astronaut, with the red dust of Mars clinging to their boots, reaches out to shake hands with an alien being, their skin a shimmering blue, under the pink-tinged sky of the fourth planet. In the background, a sleek silver rocket, a beacon of human ingenuity, stands tall, its engines powered down, as the two representatives of different worlds exchange a historic greeting amidst the desolate beauty of the Martian landscape.</div>
        </div>
        <div class="video-item">
            <video width="100%" controls>
                <source src="https://github.com/user-attachments/assets/7db2bbce-194d-434d-a605-350254b6c298" type="video/mp4">
            </video>
            <div class="caption">An elderly gentleman, with a serene expression, sits at the water's edge, a steaming cup of tea by his side. He is engrossed in his artwork, brush in hand, as he renders an oil painting on a canvas that's propped up against a small, weathered table. The sea breeze whispers through his silver hair, gently billowing his loose-fitting white shirt, while the salty air adds an intangible element to his masterpiece in progress. The scene is one of tranquility and inspiration, with the artist's canvas capturing the vibrant hues of the setting sun reflecting off the tranquil sea.</div>
        </div>
        <div class="video-item">
            <video width="100%" controls>
                <source src="https://github.com/user-attachments/assets/62b01046-8cab-44cc-bd45-4d965bb615ec" type="video/mp4">
            </video>
            <div class="caption">In a dimly lit bar, purplish light bathes the face of a mature man, his eyes blinking thoughtfully as he ponders in close-up, the background artfully blurred to focus on his introspective expression, the ambiance of the bar a mere suggestion of shadows and soft lighting.</div>
        </div>
        <div class="video-item">
            <video width="100%" controls>
                <source src="https://github.com/user-attachments/assets/d78e552a-4b3f-4b81-ac3f-3898079554f6" type="video/mp4">
            </video>
            <div class="caption">A golden retriever, sporting sleek black sunglasses, with its lengthy fur flowing in the breeze, sprints playfully across a rooftop terrace, recently refreshed by a light rain. The scene unfolds from a distance, the dog's energetic bounds growing larger as it approaches the camera, its tail wagging with unrestrained joy, while droplets of water glisten on the concrete behind it. The overcast sky provides a dramatic backdrop, emphasizing the vibrant golden coat of the canine as it dashes towards the viewer.</div>
        </div>
        <div class="video-item">
            <video width="100%" controls>
                <source src="https://github.com/user-attachments/assets/30894f12-c741-44a2-9e6e-ddcacc231e5b" type="video/mp4">
            </video>
            <div class="caption">On a brilliant sunny day, the lakeshore is lined with an array of willow trees, their slender branches swaying gently in the soft breeze. The tranquil surface of the lake reflects the clear blue sky, while several elegant swans glide gracefully through the still water, leaving behind delicate ripples that disturb the mirror-like quality of the lake. The scene is one of serene beauty, with the willows' greenery providing a picturesque frame for the peaceful avian visitors.</div>
        </div>
        <div class="video-item">
            <video width="100%" controls>
                <source src="https://github.com/user-attachments/assets/926575ca-7150-435b-a0ff-4900a963297b" type="video/mp4">
            </video>
            <div class="caption">A Chinese mother, draped in a soft, pastel-colored robe, gently rocks back and forth in a cozy rocking chair positioned in the tranquil setting of a nursery. The dimly lit bedroom is adorned with whimsical mobiles dangling from the ceiling, casting shadows that dance on the walls. Her baby, swaddled in a delicate, patterned blanket, rests against her chest, the child's earlier cries now replaced by contented coos as the mother's soothing voice lulls the little one to sleep. The scent of lavender fills the air, adding to the serene atmosphere, while a warm, orange glow from a nearby nightlight illuminates the scene with a gentle hue, capturing a moment of tender love and comfort.</div>
        </div>
    </div>
</body>
</html>

This model supports deployment using the huggingface diffusers library. You can deploy it by following these steps.

**We recommend that you visit our [GitHub](https://github.com/THUDM/CogVideo) and check out the relevant prompt optimizations and conversions to get a better experience.**

1. Install the required dependencies

```shell
# diffusers>=0.30.1
# transformers>=4.44.2
# accelerate>=0.33.0 (suggest install from source)
# imageio-ffmpeg>=0.5.1
pip install --upgrade transformers accelerate diffusers imageio-ffmpeg
```

2. Run the code

```python
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)

pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

export_to_video(video, "output.mp4", fps=8)
```

**Quantized Inference**

[PytorchAO](https://github.com/pytorch/ao) and [Optimum-quanto](https://github.com/huggingface/optimum-quanto/) can be used to quantize the Text Encoder, Transformer and VAE modules to lower the memory requirement of CogVideoX. This makes it possible to run the model on free-tier T4 Colab or smaller VRAM GPUs as well! It is also worth noting that TorchAO quantization is fully compatible with `torch.compile`, which allows for much faster inference speed.

```diff
# To get started, PytorchAO needs to be installed from the GitHub source and PyTorch Nightly.
# Source and nightly installation is only required until next release.

import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, CogVideoXPipeline
from diffusers.utils import export_to_video
+ from transformers import T5EncoderModel
+ from torchao.quantization import quantize_, int8_weight_only, int8_dynamic_activation_int8_weight

+ quantization = int8_weight_only

+ text_encoder = T5EncoderModel.from_pretrained("THUDM/CogVideoX-5b", subfolder="text_encoder", torch_dtype=torch.bfloat16)
+ quantize_(text_encoder, quantization())

+ transformer = CogVideoXTransformer3DModel.from_pretrained("THUDM/CogVideoX-5b", subfolder="transformer", torch_dtype=torch.bfloat16)
+ quantize_(transformer, quantization())

+ vae = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-5b", subfolder="vae", torch_dtype=torch.bfloat16)
+ quantize_(vae, quantization())

# Create pipeline and run inference
pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
+    text_encoder=text_encoder,
+    transformer=transformer,
+    vae=vae,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."

video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

export_to_video(video, "output.mp4", fps=8)
```

Additionally, the models can be serialized and stored in a quantized datatype to save disk space when using PytorchAO. Find examples and benchmarks at these links:
- [torchao](https://gist.github.com/a-r-r-o-w/4d9732d17412888c885480c6521a9897)
- [quanto](https://gist.github.com/a-r-r-o-w/31be62828b00a9292821b85c1017effa)

When testing using the `diffusers` library, all optimizations provided by the `diffusers` library were enabled. You can selectively disable some optimizations, including:

```
pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
```

When performing multi-GPU inference, the `enable_model_cpu_offload()` optimization needs to be disabled.
Using INT8 models will reduce inference speed.
The inference speed test also used the above VRAM optimization scheme. Without VRAM optimization, inference speed increases by about 10%. Only the `diffusers` version of the model supports quantization.

Inference Speed (Step = 50, FP/BF16): Single A100: ~180 seconds, Single H100: ~90 seconds.

**Note**

Using [SAT](https://github.com/THUDM/SwissArmyTransformer) for inference and fine-tuning of SAT version models. Feel free to visit our GitHub for more information.

---

## Factors
This section addresses variables that may impact the model's performance.

### Relevant factors:
Not available.

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
Not available.

### Motivation:
Not available.

### Preprocessing:
Not available.

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
Single GPU VRAM Consumption: <a href="https://github.com/THUDM/SwissArmyTransformer">SAT</a> BF16: 26GB, diffusers BF16: starting from 5GB*, diffusers INT8(torchao): starting from 4.4GB*.
Multi-GPU Inference VRAM Consumption: BF16: 15GB* using diffusers.
Using the `diffusers` library with all of its optimizations enabled has not been tested for actual VRAM/memory usage on devices other than **NVIDIA A100 / H100**. Generally, this solution can be adapted to all devices with **NVIDIA Ampere architecture** and above. If the optimizations are disabled, VRAM usage will increase significantly, with peak VRAM usage being about 3 times higher than the reported VRAM usage.
Using INT8 models will reduce inference speed. This is to ensure that GPUs with lower VRAM can perform inference normally while maintaining minimal video quality loss, though inference speed will decrease significantly.
[PytorchAO](https://github.com/pytorch/ao) and [Optimum-quanto](https://github.com/huggingface/optimum-quanto/) can be used to quantize the text encoder, Transformer, and VAE modules to reduce CogVideoX's memory requirements. This makes it possible to run the model on a free T4 Colab or GPUs with smaller VRAM! It is also worth noting that TorchAO quantization is fully compatible with `torch.compile`, which can significantly improve inference speed. `FP8` precision must be used on devices with `NVIDIA H100` or above, which requires installing the `torch`, `torchao`, `diffusers`, and `accelerate` Python packages from source. `CUDA 12.4` is recommended.

### Training or Fine-tuning Requirements:
Fine-tuning VRAM Consumption (per GPU): 63 GB (bs=1, LORA), 80 GB (bs=2, LORA), 75GB (bs=1, SFT).
Using [SAT](https://github.com/THUDM/SwissArmyTransformer) for fine-tuning of SAT version models.

---

## Ethical Considerations
This section discusses the ethical considerations in model development, including challenges, risks, and solutions.

Not available.

---

## Caveats and Recommendations
This section lists unresolved issues and provides guidance for users.

### Caveats:
Not available.

### Recommendations:
Not available.

---

## Additional Information
<p align="center">
  <div align="center">
  <img src=https://github.com/THUDM/CogVideo/raw/main/resources/logo.svg width="50%"/>
  </div>
</p>
<p align="center">
📍 Visit <a href="https://chatglm.cn/video?lang=en?fr=osm_cogvideo">QingYing</a> and <a href="https://open.bigmodel.cn/?utm_campaign=open&_channel_track_key=OWTVNma9">API Platform</a> to experience commercial video generation models.
</p>

## CogVideoX-2B
Inference Precision: FP16* (Recommended), BF16, FP32, FP8*, INT8, no support for INT4
Single GPU VRAM Consumption: <a href="https://github.com/THUDM/SwissArmyTransformer">SAT</a> FP16: 18GB, diffusers FP16: starting from 4GB*, diffusers INT8(torchao): starting from 3.6GB*
Multi-GPU Inference VRAM Consumption: FP16: 10GB* using diffusers
Inference Speed<br>(Step = 50, FP/BF16): Single A100: ~90 seconds<br>Single H100: ~45 seconds
Fine-tuning Precision: FP16
Fine-tuning VRAM Consumption (per GPU): 47 GB (bs=1, LORA)<br> 61 GB (bs=2, LORA)<br> 62GB (bs=1, SFT)