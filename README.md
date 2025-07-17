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
2. Run `data_collector/repo_selector.py` to order the models and select top 1000 models. The list of the top models will be saved in `top_1000_models.csv` inside the `data` directory.
```bash
python data_collector/repo_selector.py
```
3. Run `repo_readme_collector.py` to download readme files from the selected top 1000 model repositories. The readme files will be saved inside the `data/readmes` directory. Each raw readme files will be saved inside `data/readmes/raw` directory. The further processed () readme files will be saved inside `data/readmes/processed` directory.
```bash
python repo_readme_collector.py
```
4. Run `readme_selector.py` to process and select quality model cards. The selected readme files' list will be saved as `selected_repos.csv` inside `data` directory.
```bash
python readme_selector.py
```

## Reorganize model cards
Run `gemini_reorganizer.py` to reorganize the selected model cards. The reorganized model cards will be saved inside `data/reorganized` directory.
```bash
python gemini_reorganizer.py
```