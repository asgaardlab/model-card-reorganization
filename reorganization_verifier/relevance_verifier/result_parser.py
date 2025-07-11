import json

from util import path, helper

section_names = [
    'Model Details: Person or organization developing model',
    'Model Details: Model date',
    'Model Details: Model version',
    'Model Details: Model type',
    'Model Details: Training details',
    'Model Details: Paper or other resource for more information',
    'Model Details: Citation details',
    'Model Details: License',
    'Model Details: Contact',
    'Intended Use: Primary intended uses',
    'Intended Use: Primary intended users',
    'Intended Use: Out-of-scope uses',
    'How to Use',
    'Factors: Relevant factors',
    'Factors: Evaluation factors',
    'Metrics: Model performance measures',
    'Metrics: Decision thresholds',
    'Metrics: Variation approaches',
    'Evaluation Data: Datasets',
    'Evaluation Data: Motivation',
    'Evaluation Data: Preprocessing',
    'Training Data: Datasets',
    'Training Data: Motivation',
    'Training Data: Preprocessing',
    'Quantitative Analyses: Unitary results',
    'Quantitative Analyses: Intersectional results',
    'Memory or Hardware Requirements: Loading Requirements',
    'Memory or Hardware Requirements: Deploying Requirements',
    'Memory or Hardware Requirements: Training or Fine-tuning Requirements',
    'Ethical Considerations',
    'Caveats and Recommendations: Caveats',
    'Caveats and Recommendations: Recommendations'
]


def get_jury_results(model_id: str):
    model_dir_name = helper.get_repo_dir_name(model_id)
    o4_mini_result_file_path = path.LLM_RELEVANCE_VERIFICATION_RESULT_DIRECTORY / 'o4_mini' / 'run_1' / f'{model_dir_name}.json'
    try:
        with open(o4_mini_result_file_path, 'r', encoding='utf-8') as file:
            o4_mini_result = json.load(file)
    except Exception as e:
        print(f'Error reading O4 Mini result file: {e}')
        return {}

    gemini_2_5_pro_result_file_path = path.LLM_RELEVANCE_VERIFICATION_RESULT_DIRECTORY / 'gemini_2_5_pro' / 'run_1' / f'{model_dir_name}.json'
    try:
        with open(gemini_2_5_pro_result_file_path, 'r', encoding='utf-8') as file:
            gemini_2_pro_result = json.load(file)
    except Exception as e:
        print(f'Error reading Gemini 2.5 Pro result file: {e}')
        return {}

    deepseek_r1_result_file_path = path.LLM_RELEVANCE_VERIFICATION_RESULT_DIRECTORY / 'deepseek_r1' / 'run_6' / f'{model_dir_name}.json'
    try:
        with open(deepseek_r1_result_file_path, 'r', encoding='utf-8') as file:
            result = file.read()
            removed_unnecessary = result.split('</think>')[1].strip()

            if removed_unnecessary.startswith('```json'):
                removed_unnecessary = removed_unnecessary[7:-3]

            deepseek_r1_result = json.loads(removed_unnecessary)

    except Exception as e:
        print(f'Error reading DeepSeek R1 result file: {e}')
        return {}

    sections = {}
    for section_name in section_names:
        o4_mini_section_result = next(
            (section for section in o4_mini_result['sections'] if section['section_name'] == section_name), None)
        gemini_2_pro_section_result = next(
            (section for section in gemini_2_pro_result['sections'] if section['section_name'] == section_name), None)
        deepseek_r1_section_result = next(
            (section for section in deepseek_r1_result['sections'] if section['section_name'] == section_name), None)

        sections[section_name] = {
            'o4_mini_result': o4_mini_section_result,
            'gemini_2_5_pro_result': gemini_2_pro_section_result,
            'deepseek_r1_result': deepseek_r1_section_result,
        }

    return sections


def get_jury_result_mismatches(model_id: str) -> dict:
    sections = get_jury_results(model_id)
    mismatches = {}
    for section_name, section_results in sections.items():
        if section_results['o4_mini_result']['is_relevant'] != section_results['gemini_2_5_pro_result']['is_relevant'] or \
                section_results['o4_mini_result']['is_relevant'] != section_results['deepseek_r1_result']['is_relevant'] or \
                section_results['gemini_2_5_pro_result']['is_relevant'] != section_results['deepseek_r1_result']['is_relevant']:
            mismatches[section_name] = section_results

    return mismatches


if __name__ == '__main__':
    selected_models = helper.get_selected_repos()
    model_ids = selected_models['model_id'].tolist()

    mismatch_count = 0
    major_false_count = 0
    for model_id in model_ids:
        mismatches = get_jury_result_mismatches(model_id)
        print(f'------{model_id}: {len(mismatches)}')
        mismatch_count += len(mismatches)
        if mismatches:
            for section_name, mismatch in mismatches.items():
                o4_mini_relevant = mismatch['o4_mini_result']['is_relevant']
                gemini_2_pro_relevant = mismatch['gemini_2_5_pro_result']['is_relevant']
                deepseek_r1_relevant = mismatch['deepseek_r1_result']['is_relevant']
                false_count = 0
                if not o4_mini_relevant:
                    false_count += 1
                if not gemini_2_pro_relevant:
                    false_count += 1
                if not deepseek_r1_relevant:
                    false_count += 1
                if false_count >= 2:
                    major_false_count += 1
    print(f'At least 1 mismatch: {mismatch_count}')
    print(f'Major false count: {major_false_count}')