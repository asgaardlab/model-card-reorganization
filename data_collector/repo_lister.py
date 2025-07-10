from datetime import datetime
from typing import Iterable, Literal

import pandas as pd
from huggingface_hub import HfApi
from huggingface_hub.hf_api import ModelInfo

from util import path


def to_array(model: ModelInfo) -> list[Literal["auto", "manual", False] | str | int | None | datetime]:
    paper_tags = [tag for tag in model.tags if tag.startswith('arxiv:')]
    paper_ids = [paper_tag.split(':')[1] for paper_tag in paper_tags] if len(paper_tags) > 0 else []

    return [model.id, model.downloads, model.downloads_all_time, model.likes, model.created_at, model.last_modified,
            model.tags, model.sha, model.library_name, model.spaces, paper_ids]


def list_all_models() -> Iterable[ModelInfo]:
    print('Fetching model list...')
    hf_api = HfApi()
    models = hf_api.list_models(
        full=True,
        cardData=False,
        fetch_config=False
    )

    model_list = [to_array(model) for model in models]
    data_df = pd.DataFrame(model_list,
                           columns=['model_id', 'downloads', 'cumulative_downloads', 'likes', 'created_at',
                                    'last_modified', 'tags', 'sha', 'associated_library', 'model_using_spaces',
                                    'paper_ids'])

    path.ALL_MODELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data_df.to_csv(path.ALL_MODELS_FILE, index=False)

    return models


if __name__ == '__main__':
    list_all_models()
