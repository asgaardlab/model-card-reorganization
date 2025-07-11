from reorganization_verifier.consistency_verifier import lexical_similarity_calculator, \
    semantic_similarity_calculator
from reorganization_verifier.section_reader import get_section_content, get_subsection_content
from util import path, helper
from util.constants import section_subsection_headers


def subsection_level_similarity(model_id: str, run_no_1: int, run_no_2: int):
    path_1 = path.REORGANIZED_MODEL_CARD_DIRECTORY / f'run_{run_no_1}' / f'{helper.get_repo_dir_name(model_id)}.md'
    path_2 = path.REORGANIZED_MODEL_CARD_DIRECTORY / f'run_{run_no_2}' / f'{helper.get_repo_dir_name(model_id)}.md'

    lexical_similarity_scores = {}
    semantic_similarity_scores = {}
    for i, (section_header, subsection_headers) in enumerate(section_subsection_headers.items()):
        if len(subsection_headers) == 0:
            section_run_1 = get_section_content(path_1, i)
            section_run_2 = get_section_content(path_2, i)

            lexical_similarity_score = lexical_similarity_calculator.calculate_similarity(section_run_1, section_run_2)
            semantic_similarity_score = semantic_similarity_calculator.calculate_similarity(section_run_1,
                                                                                            section_run_2)

            # print(f'{section_header}: {lexical_similarity_score}, {semantic_similarity_score}')
            section_key = section_header.replace('#', '').strip()
            lexical_similarity_scores[section_key] = lexical_similarity_score
            semantic_similarity_scores[section_key] = semantic_similarity_score
        else:
            for j, subsection_header in enumerate(subsection_headers):
                subsection_run_1 = get_subsection_content(path_1, i, j)
                subsection_run_2 = get_subsection_content(path_2, i, j)

                lexical_similarity_score = lexical_similarity_calculator.calculate_similarity(subsection_run_1,
                                                                                              subsection_run_2)
                semantic_similarity_score = semantic_similarity_calculator.calculate_similarity(subsection_run_1,
                                                                                                subsection_run_2)

                # print(f'{subsection_header} {lexical_similarity_score}, {semantic_similarity_score}')
                subsection_key = f'{section_header.replace("#", "").strip()}/{subsection_header.replace("#", "").replace(":", "").strip()}'
                lexical_similarity_scores[subsection_key] = lexical_similarity_score
                semantic_similarity_scores[subsection_key] = semantic_similarity_score

    return lexical_similarity_scores, semantic_similarity_scores


def get_section_level_avg_similarity_scores(model_ids: list[str], run_no_1: int, run_no_2: int, run_no_3: int):
    avg_lexical_similarity_scores = {}
    avg_semantic_similarity_scores = {}
    for i, model_id in enumerate(model_ids):
        print(f'-------[{i + 1}] {model_id}')

        comp1_lexical_sim_score, comp1_semantic_sim_score = subsection_level_similarity(model_id, run_no_1, run_no_2)
        comp2_lexical_sim_score, comp2_semantic_sim_score = subsection_level_similarity(model_id, run_no_1, run_no_3)
        comp3_lexical_sim_score, comp3_semantic_sim_score = subsection_level_similarity(model_id, run_no_2, run_no_3)

        for subsection_header in comp1_lexical_sim_score.keys():
            # print(subsection_header, '(lexical)', comp1_lexical_sim_score[subsection_header],
            #       comp2_lexical_sim_score[subsection_header], comp3_lexical_sim_score[subsection_header])
            avg_lexical_similarity_score = (comp1_lexical_sim_score[subsection_header] +
                                            comp2_lexical_sim_score[subsection_header] +
                                            comp3_lexical_sim_score[subsection_header]) / 3
            if subsection_header not in avg_lexical_similarity_scores:
                avg_lexical_similarity_scores[subsection_header] = []
            avg_lexical_similarity_scores[subsection_header].append(avg_lexical_similarity_score)

            # print(subsection_header, '(semantic)', comp1_semantic_sim_score[subsection_header],
            #       comp2_semantic_sim_score[subsection_header], comp3_semantic_sim_score[subsection_header])
            avg_semantic_similarity_score = (comp1_semantic_sim_score[subsection_header] +
                                             comp2_semantic_sim_score[subsection_header] +
                                             comp3_semantic_sim_score[subsection_header]) / 3
            if subsection_header not in avg_semantic_similarity_scores:
                avg_semantic_similarity_scores[subsection_header] = []
            avg_semantic_similarity_scores[subsection_header].append(avg_semantic_similarity_score)

    return avg_lexical_similarity_scores, avg_semantic_similarity_scores


def document_level_similarity(model_id: str, run_no_1: int, run_no_2: int):
    path_1 = path.REORGANIZED_MODEL_CARD_DIRECTORY / f'run_{run_no_1}' / f'{helper.get_repo_dir_name(model_id)}.md'
    path_2 = path.REORGANIZED_MODEL_CARD_DIRECTORY / f'run_{run_no_2}' / f'{helper.get_repo_dir_name(model_id)}.md'

    with open(path_1, 'r', encoding='utf-8') as file:
        card_1 = file.read()
    with open(path_2, 'r', encoding='utf-8') as file:
        card_2 = file.read()

    lexical_similarity_score = lexical_similarity_calculator.calculate_similarity(card_1, card_2)
    semantic_similarity_score = semantic_similarity_calculator.calculate_similarity(card_1, card_2)

    # print(f'Full model card: {lexical_similarity_score}, {semantic_similarity_score}')
    return lexical_similarity_score, semantic_similarity_score


def get_document_level_avg_similarity_scores(model_ids: list[str], run_no_1: int, run_no_2: int, run_no_3: int):
    avg_lexical_similarity_scores = []
    avg_semantic_similarity_scores = []
    for i, model_id in enumerate(model_ids):
        print(f'-------[{i + 1}] {model_id}')

        lexical_similarity_scores = []
        semantic_similarity_scores = []

        lexical, semantic = document_level_similarity(model_id, run_no_1, run_no_2)
        lexical_similarity_scores.append(lexical)
        semantic_similarity_scores.append(semantic)

        lexical, semantic = document_level_similarity(model_id, run_no_1, run_no_3)
        lexical_similarity_scores.append(lexical)
        semantic_similarity_scores.append(semantic)

        lexical, semantic = document_level_similarity(model_id, run_no_2, run_no_3)
        lexical_similarity_scores.append(lexical)
        semantic_similarity_scores.append(semantic)

        avg_lexical_similarity_score = sum(lexical_similarity_scores) / len(lexical_similarity_scores)
        avg_semantic_similarity_score = sum(semantic_similarity_scores) / len(semantic_similarity_scores)
        print(
            f'Average similarity scores (lexical, semantic): {avg_lexical_similarity_score}, {avg_semantic_similarity_score}')

        avg_lexical_similarity_scores.append(avg_lexical_similarity_score)
        avg_semantic_similarity_scores.append(avg_semantic_similarity_score)

    return avg_lexical_similarity_scores, avg_semantic_similarity_scores
