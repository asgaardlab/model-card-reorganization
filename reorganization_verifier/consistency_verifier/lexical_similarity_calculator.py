from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from util import helper, path


def calculate_similarity(doc1: str, doc2: str) -> float:
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([doc1, doc2])

    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    return cosine_sim[0][0]


def compare_model_cards(model_id: str, run_1_no: int, run_2_no: int) -> float:
    model_card_filename = helper.get_repo_dir_name(model_id)
    run_1_file = path.REORGANIZED_MODEL_CARD_DIRECTORY / f'run_{run_1_no}' / f'{model_card_filename}.md'
    run_2_file = path.REORGANIZED_MODEL_CARD_DIRECTORY / f'run_{run_2_no}' / f'{model_card_filename}.md'

    with open(run_1_file, 'r', encoding='utf-8') as file:
        document1 = file.read()

    with open(run_2_file, 'r', encoding='utf-8') as file:
        document2 = file.read()

    return calculate_similarity(document1, document2)


def print_similarity_stats(similarity_scores: list[float]):
    average_similarity_score = sum(similarity_scores) / len(similarity_scores)
    print('avg: ', average_similarity_score)

    min_similarity_score = min(similarity_scores)
    print('min: ', min_similarity_score)

    max_similarity_score = max(similarity_scores)
    print('max: ', max_similarity_score)


if __name__ == '__main__':
    selected_models = helper.get_selected_repos()
    model_ids = selected_models['model_id'].tolist()

    first_similarity_scores = []
    second_similarity_scores = []
    for model_id in model_ids:
        first_similarity_score = compare_model_cards(model_id, 1, 2)
        first_similarity_scores.append(first_similarity_score)

        second_similarity_score = compare_model_cards(model_id, 1, 3)
        second_similarity_scores.append(second_similarity_score)

        print(model_id, first_similarity_score, second_similarity_score)

    print('run_1 vs run_2')
    print_similarity_stats(first_similarity_scores)

    print('run_1 vs run_3')
    print_similarity_stats(second_similarity_scores)
