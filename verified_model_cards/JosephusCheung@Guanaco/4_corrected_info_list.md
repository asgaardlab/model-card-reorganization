- ![](https://huggingface.co/JosephusCheung/Guanaco/resolve/main/StupidBanner.png)
- **You can run on Colab free T4 GPU now**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ocSmoy3ba1EkYu7JWT1oCw9vz8qC2cMk#scrollTo=zLORi5OcPcIJ)
- **It is highly recommended to use fp16 inference for this model.**
- **8-bit precision may significantly affect performance.**
- **If you require a more Consumer Hardware friendly version, please use the specialized quantized version.**
- **Only 5+GB V-Ram required for the specialized quantized version.**
- [JosephusCheung/GuanacoOnConsumerHardware](https://huggingface.co/JosephusCheung/GuanacoOnConsumerHardware).
- **You are encouraged to use the latest version of transformers from GitHub.**
- Guanaco is an advanced instruction-following language model.
- Guanaco is built on Meta's LLaMA 7B model.
- Guanaco expands upon the initial 52K dataset from the Alpaca model.
- An additional 534K+ entries have been incorporated into Guanaco.
- The incorporated entries cover English, Simplified Chinese, Traditional Chinese (Taiwan), Traditional Chinese (Hong Kong), Japanese, Deutsch, and various linguistic and grammatical tasks.
- This wealth of data enables Guanaco to perform exceptionally well in multilingual environments.
- The Guanaco Dataset has been made publicly accessible.
- The model weights have been released.
- The aim is to inspire more researchers to pursue related research.
- The goal is to collectively advance the development of instruction-following language models.
- [KBlueLeaf](https://huggingface.co/KBlueLeaf)’s contributions to the conceptual validation of the model are acknowledged.
- [KBlueLeaf](https://huggingface.co/KBlueLeaf)’s contributions to the [trained model](https://huggingface.co/KBlueLeaf/guanaco-7B-leh) are acknowledged.
- [KBlueLeaf](https://huggingface.co/KBlueLeaf)’s contributions to the [inference development](https://github.com/KohakuBlueleaf/guanaco-lora) of the model are acknowledged.
- The project would not have come to fruition without KBlueLeaf's efforts.
- The Guanaco model has not been filtered for harmful, biased, or explicit content.
- Outputs that do not adhere to ethical norms may be generated during use.
- Caution is advised when using the model in research or practical applications.

1. ### Improved context and prompt role support:
   - The new format is designed to be similar to ChatGPT.
   - The new format allows for better integration with the Alpaca format.
   - The new format enhances the overall user experience.
   - Instruction is utilized as a few-shot context to support diverse inputs and responses.
   - This makes it easier for the model to understand and provide accurate responses to user queries.
   - The format is as follows:
   ```
   ### Instruction:
   User: History User Input
   Assistant: History Assistant Answer
   ### Input:
   System: Knowledge
   User: New User Input
   ### Response:
   New Assistant Answer
   ```
   - This structured format allows for easier tracking of the conversation history.
   - The structured format helps maintain context throughout a multi-turn dialogue.

3. ### Role-playing support:
   - Guanaco offers advanced role-playing support, similar to Character.AI.
   - Role-playing support is available in English, Simplified Chinese, Traditional Chinese, Japanese, and Deutsch.
   - Users can instruct the model to assume specific roles, historical figures, or fictional characters.
   - Users can also instruct the model to adopt personalities based on their input.
   - This allows for more engaging and immersive conversations.
   - The model can use various sources of information to provide knowledge and context for the character's background and behavior.
   - Sources of information can include encyclopedic entries, first-person narrations, or a list of personality traits.
   - The model will consistently output responses in the format "Character Name: Reply" to maintain the chosen role throughout the conversation.

4. ### Rejection of answers and avoidance of erroneous responses:
   - The model has been updated to handle situations where it lacks sufficient knowledge or is unable to provide a valid response more effectively.
   - Reserved keywords have been introduced to indicate different scenarios.
   - The reserved keywords provide clearer communication with the user.
   - The reserved keywords can be used in the System Prompt:
     - NO IDEA: Indicates that the model lacks the necessary knowledge to provide an accurate answer.
     - The model will explain this to the user and encourage them to seek alternative sources.
     - FORBIDDEN: Indicates that the model refuses to answer due to specific reasons (e.g., legal, ethical, or safety concerns).
     - The refusal will be inferred based on the context of the query.
     - SFW: Indicates that the model refuses to answer a question because it has been filtered for NSFW content.
     - This ensures a safer and more appropriate user experience.

6. ### Continuation of responses for ongoing topics:
   - The Guanaco model can now continue answering questions or discussing topics upon the user's request.
   - This makes it more adaptable and better suited for extended conversations.
   - The contextual structure consists of System, Assistant, and User roles.
   - This structure allows the model to engage in multi-turn dialogues.
   - The structure helps maintain context-aware conversations.
   - The model can provide more coherent responses.
   - The model can accommodate role specification and character settings.
   - This provides a more immersive and tailored conversational experience based on the user's preferences.

- Guanaco is a 7B-parameter model.
- Any knowledge-based content should be considered potentially inaccurate.
- It is strongly recommended to provide verifiable sources in the System Prompt, such as Wikipedia, for knowledge-based answers.
- In the absence of sources, it is crucial to inform users of this limitation.
- This is to prevent the dissemination of false information and to maintain transparency.
- Due to the differences in the format between this project and [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca), please refer to *Guanaco-lora: LoRA for training Multilingual Instruction-following LM based on LLaMA* (https://github.com/KohakuBlueleaf/guanaco-lora) for further training and inference of our models.

## Recent News
- A recent entrant in the field is the QLoRa method.
- The QLoRa method is concerning due to its attempt to piggyback on the reputation of Guanaco.
- The project strongly disapproves of such practices.
- QLoRa lacks mathematical robustness.
- The performance of QLoRa significantly trails behind that of GPTQ and advancements such as PEFT fine-tuning.
- Guanaco has been diligent in releasing multilingual datasets since March 2023.
- Guanaco has published weights that are an enhanced version of GPTQ.
- The weights support multimodal VQA and have been optimized for 4-bit.
- The project has made a substantial financial investment of tens of thousands of dollars in distilling data from OpenAI's GPT models.
- The project considers these efforts to be incremental.
- The project aims to move beyond the incremental:
  1. The project strives to no longer rely on distillation data from OpenAI.
  2. The project has found that relying on GPT-generated data impedes significant breakthroughs.
  3. The project believes that this approach has proven to be disastrous when dealing with the imbalances in multilingual tasks.
  2. The project is focusing on the enhancement of quantization structure and partial native 4-bit fine-tuning.
  3. The project is deeply appreciative of the GPTQ-Llama project for paving the way in state-of-the-art LLM quantization.
  4. The project plans to utilize visual data to adjust its language models.
- The project believes this will fundamentally address issues of language imbalance, translation inaccuracies, and the lack of graphical logic in LLM.
- The work is still in the early stages.
- The project is determined to break new ground in these areas.
- The critique of QLoRa's practices does not stem from animosity.
- The critique is based on the belief that innovation should be rooted in originality, integrity, and substantial progress.