from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent


DATA_DIRECTORY = get_project_root() / 'data'
ALL_MODELS_FILE = DATA_DIRECTORY / 'all_models.csv'
TOP_MODELS_FILE = DATA_DIRECTORY / 'top_1000_models.csv'

README_FILE_DIRECTORY = DATA_DIRECTORY / 'readmes'
RAW_MODEL_CARD_DIRECTORY = README_FILE_DIRECTORY / 'raw'
PREPROCESSED_MODEL_CARD_DIRECTORY = README_FILE_DIRECTORY / 'preprocessed'

