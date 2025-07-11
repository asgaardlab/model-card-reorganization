import json
import shutil

from reorganization_verifier.relevance_verifier.result_parser import \
    get_jury_result_mismatches
from util import helper, path


def get_major_irrelevant_voted_mismatch(mismatches: dict) -> dict:
    major_irrelevant_voted = {}
    for mismatch_section_name, mismatch in mismatches.items():
        false_count = 0
        if mismatch['o4_mini_result']['is_relevant'] == False:
            false_count += 1
        if mismatch['gemini_2_5_pro_result']['is_relevant'] == False:
            false_count += 1
        if mismatch['deepseek_r1_result']['is_relevant'] == False:
            false_count += 1
        if false_count >= 2:
            mismatch['is_relevant'] = False
            major_irrelevant_voted[mismatch_section_name] = mismatch

    return major_irrelevant_voted


def organize_jury_result_verification_files(model_id: str):
    model_dir_name = helper.get_repo_dir_name(model_id)
    destination_dir = path.LLM_RELEVANCE_VERIFICATION_RESULT_DIRECTORY / 'jury_result_mismatch_verification' / model_dir_name

    mismatches = get_jury_result_mismatches(model_id)
    major_irrelevant_voted = get_major_irrelevant_voted_mismatch(mismatches)

    if major_irrelevant_voted:
        try:
            destination_dir.mkdir(parents=True, exist_ok=True)
            print(f'Organizing jury result verification files for model: {model_id}')

            with(destination_dir / f'{model_dir_name}_irrelevance_agreements.json').open('w', encoding='utf-8') as file:
                json.dump(major_irrelevant_voted, file, indent=2)

            reorganized_model_card_path = path.MANUAL_COMPLETENESS_VERIFICATION_RESULT_DIRECTORY / model_dir_name / f'{model_dir_name}_reorganized_4_complete.md'
            shutil.copy2(reorganized_model_card_path, destination_dir / f'{model_dir_name}_reorganized_4_complete.md')
        except Exception as e:
            print(f'Error organizing files for model {model_id}: {e}')
    return len(major_irrelevant_voted)


if __name__ == '__main__':
    selected_models = helper.get_selected_repos()
    model_ids = selected_models['model_id'].tolist()

    count = 0
    for model_id in model_ids:
        count += organize_jury_result_verification_files(model_id)
    print(count)