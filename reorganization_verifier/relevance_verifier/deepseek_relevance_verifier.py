from pathlib import Path

from api_clients.openai_chat_json_client import OpenAIChatJsonClient
from reorganization_verifier.relevance_verifier.base_relevance_verifier import \
    BaseRelevanceVerifier
from util import helper, constants


class DeepSeekRelevanceVerifier(BaseRelevanceVerifier):
    def get_save_path(self, model_id: str, next_iteration_no: int = None, sub_dir_name: str = None) -> Path:
        return super().get_save_path(model_id, next_iteration_no, 'deepseek_r1')


if __name__ == '__main__':
    deepseek_r1_client = OpenAIChatJsonClient(constants.DEEPINFRA_DEEPSEEK_R1, constants.DEEPINFRA_API_KEY, base_url=constants.DEEPINFRA_BASE_URL)
    relevance_verifier = DeepSeekRelevanceVerifier(Path('deepseek_prompt_template.md'), None, deepseek_r1_client) # json schema is added in the prompt template
    # relevance_verifier.process_single_request('CohereForAI/c4ai-command-r-plus')

    selected_models = helper.get_selected_repos()
    model_ids = selected_models['model_id'].tolist()
    relevance_verifier.process_batch_request(model_ids)