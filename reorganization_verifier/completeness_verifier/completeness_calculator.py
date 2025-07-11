from reorganization_verifier.completeness_verifier.edit_distance_calculator import \
    calculate_extra_info_removal_edit_distance, calculate_misinterpretation_removal_edit_distance, \
    calculate_missing_info_addition_edit_distance
from reorganization_verifier.completeness_verifier.missing_bullet_points_calculator import \
    count_checklist_items
from util import helper, path

if __name__ == '__main__':
    selected_models = helper.get_selected_repos()
    selected_models['model_dir_name'] = selected_models['model_id'].apply(lambda x: f'{helper.get_repo_dir_name(x)}')

    reorganization_manual_verification_dir = path.MANUAL_COMPLETENESS_VERIFICATION_RESULT_DIRECTORY

    print('word_count', 'present_points_count', 'partial_present_points', 'missing_points_count',
          'extra_info_removal_edit_distance', 'misinterpretation_removal_edit_distance',
          'missing_info_addition_edit_distance', 'model_dir_name')
    for model_verification_dir in reorganization_manual_verification_dir.iterdir():
        if model_verification_dir.is_dir():
            word_count = \
                selected_models[selected_models['model_dir_name'] == model_verification_dir.name][
                    'readme_length'].values[0]
            present_points_count, partial_present_points, missing_points_count = count_checklist_items(
                model_verification_dir)
            extra_info_removal_edit_distance = calculate_extra_info_removal_edit_distance(model_verification_dir)
            misinterpretation_removal_edit_distance = calculate_misinterpretation_removal_edit_distance(
                model_verification_dir)
            missing_info_addition_edit_distance = calculate_missing_info_addition_edit_distance(model_verification_dir)

            print(word_count, present_points_count, partial_present_points, missing_points_count,
                  extra_info_removal_edit_distance, misinterpretation_removal_edit_distance,
                  missing_info_addition_edit_distance, model_verification_dir.name)
