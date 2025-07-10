import pandas as pd

from util import path, constants


def save_top_repos() -> pd.DataFrame:
    all_models = pd.read_csv(path.ALL_MODELS_FILE)
    top_models = all_models.sort_values(by=['likes', 'downloads'], ascending=False).head(constants.TOP_N)
    top_models.to_csv(path.TOP_MODELS_FILE, index=False)
    print(f'Selected top {constants.TOP_N} models')

    return top_models


if __name__ == '__main__':
    save_top_repos()
