from sentence_transformers import SentenceTransformer, util

from util import path, helper

model = SentenceTransformer('allenai-specter', device='cpu')


def calculate_similarity(sentence1: str, sentence2: str) -> float:
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)

    cosine_similarity = util.pytorch_cos_sim(embedding1, embedding2)

    return cosine_similarity.item()


def compare_model_cards(model_id: str, run_1_no: int, run_2_no: int) -> float:
    model_card_filename = helper.get_repo_dir_name(model_id)
    run_1_file = path.REORGANIZED_MODEL_CARD_DIRECTORY / f'run_{run_1_no}' / f'{model_card_filename}.md'
    run_2_file = path.REORGANIZED_MODEL_CARD_DIRECTORY / f'run_{run_2_no}' / f'{model_card_filename}.md'
    # run_2_file = path.PREPROCESSED_MODEL_CARD_DIRECTORY / f'{model_card_filename}.md'

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
        second_similarity_score = compare_model_cards(model_id, 1, 3)
        print(model_id, first_similarity_score, second_similarity_score)

        first_similarity_scores.append(first_similarity_score)
        second_similarity_scores.append(second_similarity_score)

    print('run_14 vs run_15')
    print_similarity_stats(first_similarity_scores)

    print('run_14 vs run_16')
    print_similarity_stats(second_similarity_scores)
