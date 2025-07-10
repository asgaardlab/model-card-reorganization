import ast
import pandas as pd
from pandas import DataFrame
from util import path


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
