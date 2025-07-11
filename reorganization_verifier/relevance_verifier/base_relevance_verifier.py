from abc import ABC

from pathlib import Path

from api_clients.llm_pipeline import LLMPipeline
from util import path, helper


def get_complete_reorganized_model_card_without_additional_info_section(model_id: str) -> str:
    reorganized_model_card_path = path.MANUAL_COMPLETENESS_VERIFICATION_RESULT_DIRECTORY / f'{helper.get_repo_dir_name(model_id)}' / f'{helper.get_repo_dir_name(model_id)}_reorganized_4_complete.md'
    with open(reorganized_model_card_path, 'r', encoding='utf-8') as file:
        reorganized_model = file.read()

    marker = '## Additional Information'
    index = reorganized_model.find(marker)
    if index != -1:
        return reorganized_model[:index]
    return reorganized_model


class BaseRelevanceVerifier(LLMPipeline, ABC):
    def get_prompt(self, model_id: str) -> str:
        reorganized_model = get_complete_reorganized_model_card_without_additional_info_section(model_id)

        section_descriptions_path = path.REORGANIZER_DIRECTORY / 'model_card_template_with_description.md'
        with open(section_descriptions_path, 'r', encoding='utf-8') as file:
            section_descriptions = file.read()

        prompt = (f'Model card:\n'
                  f'"""\n'
                  f'{reorganized_model}\n'
                  f'"""\n'
                  f'\n'
                  f'Model card section descriptions:\n'
                  f'"""\n'
                  f'{section_descriptions}\n'
                  f'"""\n')
        return prompt

    def get_save_path(self, model_id: str, next_iteration_no: int = None, sub_dir_name: str = None) -> Path:
        save_root = path.LLM_RELEVANCE_VERIFICATION_RESULT_DIRECTORY / sub_dir_name

        if next_iteration_no:
            return save_root / f'run_{next_iteration_no}' / f'{helper.get_repo_dir_name(model_id)}.json'

        next_iteration_no = helper.get_next_iteration_no(save_root)
        return save_root / f'run_{next_iteration_no}' / f'{helper.get_repo_dir_name(model_id)}.json'


def remove_existings(model_ids: list, source_dir: Path) -> list:
    existing_files = [file.stem for file in source_dir.glob('*.json')]
    return [model_id for model_id in model_ids if helper.get_repo_dir_name(model_id) not in existing_files]