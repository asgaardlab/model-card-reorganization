from reorganization_verifier.section_reader import get_subsection_content, get_section_content
from util import helper, path
from util.constants import section_subsection_headers


def print_corrections(selected_models) -> None:
    for i, model_id in enumerate(selected_models['model_id'].tolist()):
        model_card_before_correction = path.MANUAL_COMPLETENESS_VERIFICATION_RESULT_DIRECTORY / f'{helper.get_repo_dir_name(model_id)}' / f'{helper.get_repo_dir_name(model_id)}_reorganized.md'
        model_card_after_correction = path.MANUAL_COMPLETENESS_VERIFICATION_RESULT_DIRECTORY / f'{helper.get_repo_dir_name(model_id)}' / f'{helper.get_repo_dir_name(model_id)}_reorganized_2_no-misinterpretation.md'

        if not model_card_before_correction.exists() or not model_card_after_correction.exists():
            # print(f'Model card files for {model_id} do not exist. Skipping...')
            continue

        print(f'----------------------------------------------------- {model_id}')
        for i, (section_header, subsection_headers) in enumerate(section_subsection_headers.items()):
            if len(subsection_headers) == 0:
                section_before_correction = get_section_content(model_card_before_correction, i)
                section_after_correction = get_section_content(model_card_after_correction, i)

                if section_before_correction != section_after_correction:
                    print(f'--- Before correction:\n{section_before_correction}')
                    print(f'--- After correction:\n{section_after_correction}')
                    print('-------------------------------------')
            else:
                for j, subsection_header in enumerate(subsection_headers):
                    subsection_before_correction = get_subsection_content(model_card_before_correction, i, j)
                    subsection_after_correction = get_subsection_content(model_card_after_correction, i, j)

                    if subsection_before_correction != subsection_after_correction:
                        print(f'--- Before correction:\n{subsection_before_correction}')
                        print(f'--- After correction:\n{subsection_after_correction}')
                        print('-------------------------------------')


def get_corrections(model_ids: list[str]) -> dict:
    mismatched_sections = {}
    for i, model_id in enumerate(model_ids):
        model_card_before_correction = path.MANUAL_COMPLETENESS_VERIFICATION_RESULT_DIRECTORY / f'{helper.get_repo_dir_name(model_id)}' / f'{helper.get_repo_dir_name(model_id)}_reorganized.md'
        model_card_after_correction = path.MANUAL_COMPLETENESS_VERIFICATION_RESULT_DIRECTORY / f'{helper.get_repo_dir_name(model_id)}' / f'{helper.get_repo_dir_name(model_id)}_reorganized_2_no-misinterpretation.md'

        if not model_card_before_correction.exists() or not model_card_after_correction.exists():
            continue

        for i, (section_header, subsection_headers) in enumerate(section_subsection_headers.items()):
            if len(subsection_headers) == 0:
                section_before_correction = get_section_content(model_card_before_correction, i)
                section_after_correction = get_section_content(model_card_after_correction, i)

                if section_before_correction != section_after_correction:
                    section_key = section_header.replace('#', '').strip()
                    if section_key not in mismatched_sections:
                        mismatched_sections[section_key] = 1
                    else:
                        mismatched_sections[section_key] += 1
            else:
                for j, subsection_header in enumerate(subsection_headers):
                    subsection_before_correction = get_subsection_content(model_card_before_correction, i, j)
                    subsection_after_correction = get_subsection_content(model_card_after_correction, i, j)

                    if subsection_before_correction != subsection_after_correction:
                        subsection_key = f'{section_header.replace("#", "").strip()}/{subsection_header.replace("#", "").replace(":", "").strip()}'
                        if subsection_key not in mismatched_sections:
                            mismatched_sections[subsection_key] = 1
                        else:
                            mismatched_sections[subsection_key] += 1
    return mismatched_sections


if __name__ == '__main__':
    selected_models = helper.get_selected_repos()
    # print_corrections(selected_models)
    print(get_corrections(selected_models['model_id'].tolist()))