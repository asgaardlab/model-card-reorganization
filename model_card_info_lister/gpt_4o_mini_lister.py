from pathlib import Path

from api_clients.llm_pipeline import LLMPipeline
from api_clients.openai_chat_client import OpenAIChatClient
from util import helper, path, constants


class ModelCardInfoLister(LLMPipeline):
    def get_prompt(self, model_id: str) -> str:
        model_card = helper.get_model_card(model_id)
        user_message = ('Here is the model card:'
                        f'```md'
                        f'{model_card}'
                        f'```')
        return user_message

    def get_save_path(self, model_id: str, next_iteration_no: int = None) -> Path:
        save_root = path.MODEL_CARD_INFO_LIST_DIRECTORY
        next_iteration_no = helper.get_next_iteration_no(save_root)
        return save_root / f'run_{next_iteration_no}' / f'{helper.get_repo_dir_name(model_id)}.md'


if __name__ == '__main__':
    gpt_4o_mini_chat_client = OpenAIChatClient(constants.GPT_4O_MINI, constants.OPENAI_API_KEY)
    info_lister = ModelCardInfoLister(Path('system_instruction.md'), None, gpt_4o_mini_chat_client)
    # info_lister.make_single_request('WizardLMTeam/WizardCoder-15B-V1.0')

    all_selected_repos = helper.get_selected_repos()
    info_lister.process_batch_request(all_selected_repos['model_id'].tolist())
