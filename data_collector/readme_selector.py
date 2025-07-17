import pandas as pd

from data_collector.roberta_language_detector import detect_readmes_language
from util import constants, helper, path


def detect_readmes_length():
    top_models = helper.get_top_model_list(constants.TOP_N)
    top_models['readme_length'] = top_models['model_id'].apply(lambda model_id: helper.get_readme_length(model_id))
    helper.save_top_model_list(top_models)


def is_base_model(model_id: str) -> bool:
    raw_readme_path = path.RAW_MODEL_CARD_DIRECTORY / f'{helper.get_repo_dir_name(model_id)}.md'
    if raw_readme_path.exists():
        with open(raw_readme_path, 'r', encoding='utf-8') as readme_file:
            raw_readme = readme_file.read()
            yaml_metadata = raw_readme.split('---')[1] if raw_readme.startswith('---') else ''
            if 'base_model:' in yaml_metadata:
                return False
            else:
                return True
    return False


def detect_if_base_model():
    top_models = helper.get_top_model_list(constants.TOP_N)
    top_models['is_base_model'] = top_models['model_id'].apply(lambda model_id: is_base_model(model_id))
    helper.save_top_model_list(top_models)


def select_repos(top_n:int, length_threshold: int):
    top_models = helper.get_top_model_list(top_n)

    top_models_with_english_readme = top_models[top_models['model_card_language_roberta'] == 'en']
    print(f'Selected models with English readme: {len(top_models_with_english_readme)}')

    models_with_long_readmes = top_models_with_english_readme[top_models_with_english_readme['readme_length'] > length_threshold]
    print(f'Selected models with at least {length_threshold} words long readme: {len(models_with_long_readmes)}')

    base_models = models_with_long_readmes[models_with_long_readmes['is_base_model']]
    print(f'Selected base models: {len(base_models)}')

    base_models['organization_id'] = base_models['model_id'].apply(lambda model_id: model_id.split('/')[0])
    base_models.sort_values(by=['organization_id', 'likes', 'downloads'], ascending=[True, False, False], inplace=True)
    one_model_per_organization = base_models.groupby('organization_id').head(1)
    print(f'Selected top model from each organization: {len(one_model_per_organization)}')

    one_model_per_organization.to_csv(path.TOP_ONE_MODEL_PER_ORGANIZATION_FILE, index=False)


def exclude_unwanted_repos():
    top_models = pd.read_csv(path.TOP_ONE_MODEL_PER_ORGANIZATION_FILE)
    excluding_repos = pd.read_csv(path.EXCLUDING_REPOS_FILE)

    selected_models = top_models[~top_models['model_id'].isin(excluding_repos['model_id'])]
    selected_models.to_csv(path.SELECTED_REPOS_FILE, index=False)
    print(f'Selected models after removing unwanted ones: {len(selected_models)}')


if __name__ == '__main__':
    detect_readmes_language()
    detect_readmes_length()
    detect_if_base_model()

    select_repos(constants.TOP_N, 1000)
    print('Manually verify models listed in "data/top_one_model_per_organization.csv" and list the unwanted models in "data/excluding_repos.csv". If you don\'t have any unwanted models, just leave it empty.')
