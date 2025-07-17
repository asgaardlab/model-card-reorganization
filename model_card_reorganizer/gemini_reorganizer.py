from pathlib import Path

from api_clients.llm_pipeline import LLMPipeline
from api_clients.vertexai_client import VertexAIClient
from util import path, helper, constants


class GeminiModelCardReorganizer(LLMPipeline):
    def get_prompt(self, model_id: str) -> str:
        with open(path.REORGANIZER_DIRECTORY / 'gemini_prompt_template.md', 'r') as file:
            prompt_template = file.read()

        with open(path.REORGANIZER_DIRECTORY / 'model_card_template_with_description.md', 'r') as file:
            model_card_template = file.read()

        prompt = (f'{prompt_template}'
                  f''
                  f'**Model Card Content:**'
                  f'"""'
                  f'{helper.get_model_card(model_id)}'
                  f'"""'
                  f''
                  f'**Model Card Template:**'
                  f'"""'
                  f'{model_card_template}'
                  f'"""'
                  f''
                  f'**Reorganized Model Card:**')
        return prompt

    def get_save_path(self, model_id: str, next_iteration_no = None) -> Path:
        save_root = path.REORGANIZED_MODEL_CARD_DIRECTORY
        if next_iteration_no:
            return save_root / f'run_{next_iteration_no}' / f'{helper.get_repo_dir_name(model_id)}.md'
        next_iteration_no = helper.get_next_iteration_no(save_root)
        return save_root / f'run_{next_iteration_no}' / f'{helper.get_repo_dir_name(model_id)}.md'


if __name__ == '__main__':
    gemini_client = VertexAIClient(constants.GEMINI_2_FLASH_THINKING, constants.GEMINI_API_KEY)
    reorganizer = GeminiModelCardReorganizer(None, None, gemini_client)
    # reorganizer.process_single_request('HuggingFaceM4/idefics2-8b')

    selected_models = helper.get_selected_repos()
    model_ids = selected_models['model_id'].tolist()
    reorganizer.process_batch_request(model_ids)
