import re
from pathlib import Path

from util import helper, path


def count_checklist_items(model_dir: Path) -> tuple[int, int, int]:
    info_list_path_pattern = '*_info_list.md'
    info_list_path = list(model_dir.glob(info_list_path_pattern))[0]

    with open(info_list_path, 'r', encoding='utf-8') as file:
        info_list = file.read()

    present_items_count = len(re.findall(r"\nV- ", info_list))
    partial_present_items_count = len(re.findall(r"\nP- ", info_list))
    missing_items_count = len(re.findall(r"\nX- ", info_list))

    return present_items_count, partial_present_items_count, missing_items_count


def get_missing_checklist_items(model_dir: Path) -> list[str]:
    info_list_path_pattern = '*_info_list.md'
    info_list_path = list(model_dir.glob(info_list_path_pattern))[0]

    with open(info_list_path, 'r', encoding='utf-8') as file:
        info_list = file.read()

    matches = re.finditer(r"\nX- ", info_list)

    # find the next V- or P- point after each X- point
    missing_items = []
    for match in matches:
        search_start_index = match.start() + 3  # +3 to skip the 'X- ' part
        x_end_index = info_list.find('\nX- ', search_start_index)
        v_end_index = info_list.find('\nV- ', search_start_index)
        p_end_index = info_list.find('\nP- ', search_start_index)

        indexes = [x_end_index, v_end_index, p_end_index, len(info_list)]
        filtered_min = min([v for v in indexes if v != -1])

        missing_item = info_list[search_start_index - 3:filtered_min].strip()
        missing_items.append(missing_item)

    return missing_items


if __name__ == '__main__':
    selected_models = helper.get_selected_repos()

    for i, model_id in enumerate(selected_models['model_id'].tolist()):
        model_dir_path = path.MANUAL_COMPLETENESS_VERIFICATION_RESULT_DIRECTORY / helper.get_repo_dir_name(model_id)

        if not model_dir_path.exists():
            print(f'[{i+1}] --- Skipping {model_id}')
            continue

        print(f'[{i+1}] --- Analyzing {model_id}')

        missing_checklist_items = get_missing_checklist_items(model_dir_path)
        for item in missing_checklist_items:
            print(f'{item}')

    # model_dir_path = path.MANUAL_COMPLETENESS_VERIFICATION_RESULT_DIRECTORY / helper.get_repo_dir_name('CAMB-AI/MARS5-TTS')
    # missing_checklist_items = get_missing_checklist_items(model_dir_path)
    # for item in missing_checklist_items:
    #     print(f'{item}')