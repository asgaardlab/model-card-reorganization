import pandas as pd
import seaborn as sns
import matplotlib.ticker as mtick
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from reorganization_verifier.completeness_verifier.edit_distance_calculator import \
    calculate_missing_info_addition_edit_distance, calculate_incorrect_info_removal_edit_distance
from reorganization_verifier.completeness_verifier.missing_bullet_points_calculator import \
    count_checklist_items
from reorganization_verifier.consistency_verifier.consistency_calculator import \
    get_document_level_avg_similarity_scores
from util import helper, path


def visualize_and_save_distribution(distribution: list[float], save_file_name: str) -> None:
    fig = plt.figure(figsize=(5, 1.15))
    with sns.axes_style('white'):
        fig.add_subplot()
        ax = sns.violinplot(x=distribution, orient='h', color='C0')
        sns.despine(ax=ax, top=True, right=True, left=False)

    plt.xlabel('')
    plt.xlim(0.0, 1.0)  # 0–100% range
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    fig.tight_layout()

    path.GRAPH_DIRECTORY.mkdir(parents=True, exist_ok=True)
    plt.savefig(path.GRAPH_DIRECTORY / save_file_name)

    plt.show()


def visualize_missing_ratio_percentage(total_items: list[int], missing_items: list[int]) -> None:
    data = {
        'total_items': total_items,
        'missing_items': missing_items
    }

    df = pd.DataFrame(data)
    df['missing_ratio'] = df['missing_items'] / df['total_items']
    print(df)

    avg = df['missing_ratio'].mean()
    med = df['missing_ratio'].median()
    std = df['missing_ratio'].std()
    min_ = df['missing_ratio'].min()
    max_ = df['missing_ratio'].max()

    print(f'Across the model cards, the average missing ratio was {avg:.1%} '
          f'(median: {med:.1%}, SD: {std:.1%}). '
          f'The best-performing model card had {min_:.1} missing items, '
          f'while the worst had {max_:.1%} of items missing.')

    correlation = df['total_items'].corr(df['missing_items'])
    print(f"Pearson correlation between total and missing checklist items: {correlation:.2f}")

    r, p_value = pearsonr(df['total_items'], df['missing_items'])
    print(f"Correlation: {r:.2f}, p-value: {p_value:.4f}")

    fig = plt.figure(figsize=(5, 1.15))
    with sns.axes_style('white'):
        fig.add_subplot()
        ax = sns.violinplot(x=df['missing_ratio'], orient='h', color='C0')
        sns.despine(ax=ax, top=True, right=True, left=False)

    plt.xlabel('')
    plt.xlim(0, 1)  # 0–100% range
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    fig.tight_layout()

    path.GRAPH_DIRECTORY.mkdir(parents=True, exist_ok=True)
    plt.savefig(path.GRAPH_DIRECTORY / 'missing_points_percentage_distribution.pdf')

    plt.show()


def visualize_edit_distance_distribution(edit_distances: list[float], save_file_name: str) -> None:
    edit_distances_df = pd.DataFrame(edit_distances)
    # print(edit_distances_df)

    avg = edit_distances_df[0].mean()
    med = edit_distances_df[0].median()
    std = edit_distances_df[0].std()
    min_ = edit_distances_df[0].min()
    max_ = edit_distances_df[0].max()

    print(f'Min: {min_}, Max: {max_}, Avg: {avg}, Median: {med}, Std: {std}')

    visualize_and_save_distribution(edit_distances, save_file_name)


def visualize_document_level_similarity_scores(model_ids: list[str]):
    avg_lexical_scores, avg_semantic_scores = get_document_level_avg_similarity_scores(
        model_ids, 1, 2, 3)
    visualize_and_save_distribution(avg_lexical_scores, 'document_level_lexical_similarity_scores_distribution.pdf')
    visualize_and_save_distribution(avg_semantic_scores, 'document_level_semantic_similarity_scores_distribution.pdf')


if __name__ == '__main__':
    selected_models = helper.get_selected_repos()
    selected_models['model_dir_name'] = selected_models['model_id'].apply(lambda x: f'{helper.get_repo_dir_name(x)}')

    total_items = []
    missing_items = []
    completeness_edit_distances = []
    correction_edit_distances = []
    for model_verification_dir in path.MANUAL_COMPLETENESS_VERIFICATION_RESULT_DIRECTORY.iterdir():
        if model_verification_dir.is_dir():
            present_items_count, partial_present_items_count, missing_items_count = count_checklist_items(
                model_verification_dir)
            total_items_count = present_items_count + partial_present_items_count + missing_items_count
            total_items.append(total_items_count)
            missing_items.append(missing_items_count)
            completeness_edit_distances.append(calculate_missing_info_addition_edit_distance(model_verification_dir))
            correction_edit_distances.append(calculate_incorrect_info_removal_edit_distance(model_verification_dir))

    visualize_missing_ratio_percentage(total_items, missing_items)
    print('Visualizing missing info addition edit distance ratio distribution...')
    visualize_edit_distance_distribution(completeness_edit_distances, 'completeness_edit_distance_distribution.pdf')
    print('Visualizing correction edit distance ratio distribution...')
    visualize_edit_distance_distribution(correction_edit_distances, 'correction_edit_distance_distribution.pdf')
    print('Visualizing document-level similarity scores...')
    visualize_document_level_similarity_scores(selected_models['model_id'].tolist())