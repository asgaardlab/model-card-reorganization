# model-card-reorganization
## Setup environment
Python 3.11. Did not check backward compatibility
1. Install the required packages
```bash
pip install -r requirements.txt
```

## Select model cards from Hugging Face repositories
1. Run `data_collector/repo_lister.py` to list all the models available in Hugging Face. A file named `all_models.csv` with the model list will be created  inside the `data` directory.
```bash
python data_collector/repo_lister.py
```
2. Run `data_collector/repo_selector.py` to order the models and select top 1000 models. The list of the top models will be saved in `data/top_1000_models.csv`.
```bash
python data_collector/repo_selector.py
```
3. Run `data_collector/repo_readme_collector.py` to download readme files of the selected top 1000 models. The readme files will be saved inside the `data/readmes` directory. Each raw readme files will be saved inside `data/readmes/raw` directory. The further processed readme files will be saved inside `data/readmes/processed` directory.
```bash
python data_collector/repo_readme_collector.py
```
4. Run `data_collector/readme_selector.py` to process and select automated quality model cards. The list will be saved in `data/top_one_model_per_organization.csv`.
```bash
python data_collector/readme_selector.py
```
5. Manually verify models listed in `data/top_one_model_per_organization.csv` and list the unwanted models in `data/excluding_repos.csv`. If you don't have any unwanted models, just leave it empty with a `model_id` as header of the file. Now, Run `data_collector/exclude_unwamted_repos.py` to get the final selected list of quality model cards saved in `data/selected_repos.csv`.
```bash
python data_collector/exclude_unwanted_repos.py
```

## Reorganize model cards
Run `gemini_reorganizer.py` to reorganize the selected model cards. The reorganized model cards will be saved inside `data/reorganized` directory.
```bash
python gemini_reorganizer.py
```