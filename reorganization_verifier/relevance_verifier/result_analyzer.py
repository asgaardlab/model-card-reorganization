import collections
import json

from util import path, helper


def get_irrelevance_subsections(model_ids: list[str]) -> list[str]:
    manual_verification_results = path.LLM_RELEVANCE_VERIFICATION_RESULT_DIRECTORY / 'jury_result_mismatch_verification'
    section_names = []
    for i, model_id in enumerate(model_ids):
        model_dir_name = helper.get_repo_dir_name(model_id)
        model_dir_path = manual_verification_results / model_dir_name

        if not model_dir_path.exists():
            print(f'[{i+1}] --- Skipping {model_dir_name}')
            continue

        print(f'[{i+1}] --- Analyzing {model_dir_name}')

        verification_file_path_pattern = '*_irrelevance_agreements.json'
        verification_file_path = list(model_dir_path.glob(verification_file_path_pattern))[0]

        with verification_file_path.open('r', encoding='utf-8') as file:
            verification_data = json.load(file)

        for section_name, section_data in verification_data.items():
            if section_data['is_relevant'] is False:
                section_key = section_name.replace(': ', '/').strip()
                section_names.append(section_key)
    return section_names


if __name__ == '__main__':
    selected_models = helper.get_selected_repos()
    manual_verification_results = path.LLM_RELEVANCE_VERIFICATION_RESULT_DIRECTORY / 'jury_result_mismatch_verification'

    total_count = 0
    total_is_relevant_false = 0
    section_names = []
    for i, model_id in enumerate(selected_models['model_id'].tolist()):
        model_dir_name = helper.get_repo_dir_name(model_id)
        model_dir_path = manual_verification_results / model_dir_name

        if not model_dir_path.exists():
            print(f'[{i+1}] --- Skipping {model_dir_name}')
            continue

        print(f'[{i+1}] --- Analyzing {model_dir_name}')

        verification_file_path_pattern = '*_irrelevance_agreements.json'
        verification_file_path = list(model_dir_path.glob(verification_file_path_pattern))[0]

        with verification_file_path.open('r', encoding='utf-8') as file:
            verification_data = json.load(file)

        for section_name, section_data in verification_data.items():
            total_count += 1
            if section_data['is_relevant'] is False:
                total_is_relevant_false += 1
                section_key = section_name.replace(': ', '/').strip()
                section_names.append(section_key)
            else:
                pretty_json = json.dumps(section_data, indent=4)
                print(pretty_json)

    print(f'Total (sub)sections analyzed: {total_count}')
    print(f'Total (sub)sections with is_relevant = False: {total_is_relevant_false}')
    print(collections.Counter(section_names))