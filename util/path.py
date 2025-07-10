from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent


DATA_DIRECTORY = get_project_root() / 'data'
ALL_MODELS_FILE = DATA_DIRECTORY / 'all_models.csv'
