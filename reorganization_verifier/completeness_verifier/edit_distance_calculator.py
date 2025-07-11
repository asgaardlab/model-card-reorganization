from Levenshtein import distance
from pathlib import Path


def calculate_edit_distance(source_path: Path, target_path: Path) -> int:
    with open(source_path, 'r', encoding='utf-8') as file:
        source = file.read()

    with open(target_path, 'r', encoding='utf-8') as file:
        target = file.read()

    return distance(source, target)


def calculate_normalized_edit_distance(source_path: Path, target_path: Path) -> float:
    with open(source_path, 'r', encoding='utf-8') as file:
        source = file.read()

    with open(target_path, 'r', encoding='utf-8') as file:
        target = file.read()

    levenshtein_distance = distance(source, target)
    max_len = max(len(source), len(target))

    return (levenshtein_distance / max_len) if max_len > 0 else 0.0


def calculate_extra_info_removal_edit_distance(model_dir: Path) -> int:
    reorganized_path_pattern = '*_reorganized.md'
    reorganized_path = list(model_dir.glob(reorganized_path_pattern))[0]

    extra_removed_path_pattern = '*_1_no-extra.md'
    extra_removed_path = list(model_dir.glob(extra_removed_path_pattern))[0]

    return calculate_edit_distance(reorganized_path, extra_removed_path)


def calculate_misinterpretation_removal_edit_distance(model_dir: Path) -> int:
    extra_removed_path_pattern = '*_1_no-extra.md'
    extra_removed_path = list(model_dir.glob(extra_removed_path_pattern))[0]

    misinterpretation_removed_path_pattern = '*_2_no-misinterpretation.md'
    misinterpretation_removed_path = list(model_dir.glob(misinterpretation_removed_path_pattern))[0]

    return calculate_edit_distance(extra_removed_path, misinterpretation_removed_path)


def calculate_incorrect_info_removal_edit_distance(model_dir: Path) -> float:
    reorganized_path_pattern = '*_reorganized.md'
    reorganized_path = list(model_dir.glob(reorganized_path_pattern))[0]

    incorrect_info_removed_path_pattern = '*_2_no-misinterpretation.md'
    misinterpretation_removed_path = list(model_dir.glob(incorrect_info_removed_path_pattern))[0]

    return calculate_normalized_edit_distance(reorganized_path, misinterpretation_removed_path)


def calculate_missing_info_addition_edit_distance(model_dir: Path) -> float:
    misinterpretation_removed_path_pattern = '*_2_no-misinterpretation.md'
    misinterpretation_removed_path = list(model_dir.glob(misinterpretation_removed_path_pattern))[0]

    missing_info_added_path_pattern = '*_3_add-missing.md'
    missing_info_added_path = list(model_dir.glob(missing_info_added_path_pattern))[0]

    return calculate_normalized_edit_distance(misinterpretation_removed_path, missing_info_added_path)