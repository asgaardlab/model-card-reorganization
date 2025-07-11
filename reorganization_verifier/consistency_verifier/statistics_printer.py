import collections
import statistics
from collections import Counter

from model_card_reorganization_verifier.accuracy_verifier.section_comparer import get_corrections
from model_card_reorganization_verifier.consistency_verifier.consistency_calculator import \
    get_document_level_avg_similarity_scores, get_section_level_avg_similarity_scores
from model_card_reorganization_verifier.correctness_verifier.relevance_verifier.result_analyzer import \
    get_irrelevance_subsections
from util import helper
from util.helper import round_half_up, section_subsection_headers


def print_document_level_statistics(model_ids, run_no_1, run_no_2, run_no_3) -> None:
    avg_lexical_scores, avg_semantic_scores = get_document_level_avg_similarity_scores(
        model_ids, run_no_1, run_no_2, run_no_3)

    print(f'{len(avg_lexical_scores)} document level average lexical similarity scores')
    average_similarity_score = sum(avg_lexical_scores) / len(avg_lexical_scores)
    print('avg: ', average_similarity_score)

    min_similarity_score = min(avg_lexical_scores)
    print('min: ', min_similarity_score)

    max_similarity_score = max(avg_lexical_scores)
    print('max: ', max_similarity_score)

    median_similarity_score = statistics.median(avg_lexical_scores)
    print(f"Median: {median_similarity_score}")

    print(f'{len(avg_semantic_scores)} document level average semantic similarity scores')
    average_similarity_score = sum(avg_semantic_scores) / len(avg_semantic_scores)
    print('avg: ', average_similarity_score)

    min_similarity_score = min(avg_semantic_scores)
    print('min: ', min_similarity_score)

    max_similarity_score = max(avg_semantic_scores)
    print('max: ', max_similarity_score)

    median_similarity_score = statistics.median(avg_semantic_scores)
    print(f'median: {median_similarity_score}')


def print_section_level_statistics(model_ids, run_no_1, run_no_2, run_no_3) -> None:
    avg_lexical_scores_per_section, avg_semantic_scores_per_section = get_section_level_avg_similarity_scores(
        model_ids, run_no_1, run_no_2, run_no_3)

    print('\t\\begin{tabular}{lrrrr}')
    print('\t\t\\toprule')
    print(
        '\t\t\multirow{2}{*}{\\textbf{(Sub)sections}} & \multicolumn{2}{c}{\\textbf{Semantic similarity scores}} & \multicolumn{2}{c}{\\textbf{Lexical similarity scores}} \\\\')
    print('\t\t\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}')
    print('\t\t & \\textbf{Avg.} & \\textbf{Median} & \\textbf{Avg.} & \\textbf{Median} \\\\')
    print('\t\t\\midrule')
    for section_header, avg_lexical_scores in avg_lexical_scores_per_section.items():
        avg_lexical_similarity_score = sum(avg_lexical_scores) / len(avg_lexical_scores)
        median_lexical_similarity_score = statistics.median(avg_lexical_scores)

        avg_semantic_scores = avg_semantic_scores_per_section[section_header]
        avg_semantic_similarity_score = sum(avg_semantic_scores) / len(avg_semantic_scores_per_section[section_header])
        median_semantic_similarity_score = statistics.median(avg_semantic_scores)

        print(
            f'\t\t{section_header} & {round_half_up(avg_semantic_similarity_score)} & {round_half_up(median_semantic_similarity_score)} & {round_half_up(avg_lexical_similarity_score)} & {round_half_up(median_lexical_similarity_score)} \\\\')
    print('\t\t\\bottomrule')
    print('\t\end{tabular}')


def print_section_level_merged_table(model_ids, run_no_1, run_no_2, run_no_3) -> None:
    corrections_count = get_corrections(selected_models['model_id'].tolist())

    irrelevance_subsections = get_irrelevance_subsections(model_ids)
    irrelevance_per_subsection_count = collections.Counter(irrelevance_subsections)

    avg_lexical_scores_per_subsection, avg_semantic_scores_per_subsection = get_section_level_avg_similarity_scores(
        model_ids, run_no_1, run_no_2, run_no_3)

    print('\t\\begin{tabular}{@{}l@{}rrrrrr@{}}')
    print('\t\t\\toprule')
    print(
        '\t\t\multirow{3}{*}{\\textbf{(Sub)sections}} & \multirow{3}{*}{\shortstack{\\textbf{\# of} \\\\ \\textbf{corrections} \\\\ \\textbf{(out of 48)}}} & \multirow{3}{*}{\shortstack{\\textbf{\# of} \\\\ \\textbf{irrelevance} \\\\ \\textbf{(out of 48)}}} & \multicolumn{2}{c}{\\textbf{Semantic}} & \multicolumn{2}{c}{\\textbf{Lexical}} \\\\')
    print(
        '\t\t & & & \multicolumn{2}{c}{\\textbf{similarity}} & \multicolumn{2}{c}{\\textbf{similarity}} \\\\')
    print('\t\t\\cmidrule(lr){4-5} \\cmidrule(lr){6-7}')
    print('\t\t & & & \\textbf{Avg.} & \\textbf{Med.} & \\textbf{Avg.} & \\textbf{Med.} \\\\')
    print('\t\t\\midrule')

    for i, (section_header, subsection_headers) in enumerate(section_subsection_headers.items()):
        if len(subsection_headers) == 0:
            key = section_header.replace('#', '').strip()
            print_row(corrections_count, irrelevance_per_subsection_count, avg_semantic_scores_per_subsection, avg_lexical_scores_per_subsection, key)
        else:
            for j, subsection_header in enumerate(subsection_headers):
                key = f'{section_header.replace("#", "").strip()}/{subsection_header.replace("#", "").replace(":", "").strip()}'
                print_row(corrections_count, irrelevance_per_subsection_count, avg_semantic_scores_per_subsection, avg_lexical_scores_per_subsection, key)

    print('\t\t\\bottomrule')
    print('\t\end{tabular}')


def print_row(corrections_count, irrelevance_per_subsection_count, avg_semantic_scores_per_subsection, avg_lexical_scores_per_subsection, key):
    correction_count = corrections_count[key] if key in corrections_count else 0
    correction_percentage = (correction_count / 48) * 100

    irrelevance_count = irrelevance_per_subsection_count[key] if key in irrelevance_per_subsection_count else 0
    irrelevance_percentage = (irrelevance_count / 48) * 100

    avg_semantic_scores = avg_semantic_scores_per_subsection[key]
    avg_semantic_similarity_score = sum(avg_semantic_scores) / len(avg_semantic_scores)
    median_semantic_similarity_score = statistics.median(avg_semantic_scores)

    avg_lexical_scores = avg_lexical_scores_per_subsection[key]
    avg_lexical_similarity_score = sum(avg_lexical_scores) / len(avg_lexical_scores)
    median_lexical_similarity_score = statistics.median(avg_lexical_scores)

    subsection_avg_semantic_similarity_score = round_half_up(avg_semantic_similarity_score)
    subsection_median_semantic_similarity_score = round_half_up(median_semantic_similarity_score)
    subsection_avg_lexical_similarity_score = round_half_up(avg_lexical_similarity_score)
    subsection_median_lexical_similarity_score = round_half_up(median_lexical_similarity_score)

    print(
        f'\t\t{key} & '
        f'\cellcolor{{niceorange!{correction_percentage}!white}}{correction_count} ({correction_percentage:.2f}\%) & '
        f'\cellcolor{{niceorange!{irrelevance_percentage}!white}}{irrelevance_count} ({irrelevance_percentage:.2f}\%) & '
        f'\cellcolor{{softc0!{subsection_avg_semantic_similarity_score * 100}!white}}{subsection_avg_semantic_similarity_score} & '
        f'\cellcolor{{softc0!{subsection_median_semantic_similarity_score * 100}!white}}{subsection_median_semantic_similarity_score} & '
        f'\cellcolor{{softc0!{subsection_avg_lexical_similarity_score * 100}!white}}{subsection_avg_lexical_similarity_score} & '
        f'\cellcolor{{softc0!{subsection_median_lexical_similarity_score * 100}!white}}{subsection_median_lexical_similarity_score} \\\\')


if __name__ == '__main__':
    selected_models = helper.get_selected_repos()
    run_1 = 14
    run_2 = 15
    run_3 = 16

    # print_document_level_statistics(selected_models['model_id'].tolist(), run_1, run_2, run_3)
    # print_section_level_statistics(selected_models['model_id'].tolist(), run_1, run_2, run_3)
    print_section_level_merged_table(selected_models['model_id'].tolist(), run_1, run_2, run_3)
