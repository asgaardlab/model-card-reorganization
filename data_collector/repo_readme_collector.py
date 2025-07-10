import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from data_collector.readme_processor import process_readme_files
from util import path, helper, constants
from util.helper import get_repo_dir_name


def organize_downloaded_readme_file(repo_dir: Path):
    readme_file = repo_dir / 'README.md'
    if readme_file.exists():
        rename_to = repo_dir / f'{repo_dir.name}.md'
        os.rename(readme_file, rename_to)
        shutil.move(rename_to, path.RAW_MODEL_CARD_DIRECTORY)
    shutil.rmtree(repo_dir)


def download_readme_file(model_id: str, commit_hash: str):
    clone_to = path.RAW_MODEL_CARD_DIRECTORY / get_repo_dir_name(model_id)

    readme_file_path = path.RAW_MODEL_CARD_DIRECTORY / f'{get_repo_dir_name(model_id)}.md'
    if readme_file_path.exists():
        return

    clone_to.mkdir(parents=True, exist_ok=True)
    try:
        hf_hub_download(repo_id=model_id, revision=commit_hash, filename='README.md', local_dir=clone_to)
        organize_downloaded_readme_file(clone_to)
    except Exception as e:
        print(f'Error: {model_id} - {e}')
        shutil.rmtree(clone_to)


def download_top_repos_readme_file():
    top_models = helper.get_top_model_list(constants.TOP_N)

    print(f'Downloading readme files for top {constants.TOP_N} models')
    top_models.apply(lambda model: download_readme_file(model['model_id'], model['sha']), axis=1)

    print(f'{sum(1 for file in path.RAW_MODEL_CARD_DIRECTORY.iterdir() if file.is_file())} repositories have model card')


def download_and_process_readme_files():
    download_top_repos_readme_file()
    process_readme_files()


if __name__ == '__main__':
    download_and_process_readme_files()
