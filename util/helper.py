import ast
import pandas as pd
from pandas import DataFrame
from util import path
from util.md_processor import remove_codeblock


def get_top_model_list(top_n: int = 1000) -> DataFrame:
    if top_n > 1000:
        raise ValueError('Top n should be less than or equal to 10000')

    top_models = pd.read_csv(path.TOP_MODELS_FILE)

    if 'paper_ids' in top_models.columns:
        top_models['paper_ids'] = top_models['paper_ids'].apply(ast.literal_eval)

    if 'repo_file_names' in top_models.columns:
        top_models['repo_file_names'] = top_models['repo_file_names'].apply(ast.literal_eval)

    return top_models.head(top_n)


def get_repo_dir_name(model_id: str) -> str:
    return model_id.replace("/", "@")


def get_readme_length(model_id: str) -> int:
    readme_path = path.PREPROCESSED_MODEL_CARD_DIRECTORY / f'{get_repo_dir_name(model_id)}.md'

    if not readme_path.exists():
        return 0

    with open(readme_path, 'r', encoding='utf-8') as readme_file:
        readme = readme_file.read()

    readme = remove_codeblock(readme)
    readme_words = readme.split()
    return len(readme_words)


def save_top_model_list(top_models: DataFrame) -> None:
    top_models.to_csv(path.TOP_MODELS_FILE, index=False)


def get_model_card(model_id: str) -> str:
    model_card_path = path.PREPROCESSED_MODEL_CARD_DIRECTORY / f'{get_repo_dir_name(model_id)}.md'

    if not model_card_path.exists():
        raise FileNotFoundError(f'No model card found for {model_id}')

    with open(model_card_path, 'r', encoding='utf-8') as f:
        return f.read().strip()
