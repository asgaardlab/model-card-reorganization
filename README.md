# model-card-reorganization
Python 3.11. Did not check backward compatibility

1. Install the required packages
```bash
pip install -r requirements.txt
```

1. Run `repo_lister.py` to list all the model cards in the repository. The file will be saved as `all_models.csv` inside the `data` directory.
```bash
python repo_lister.py
```
2. Run `repo_selector.py` to select top 1000 models. The list of models will be saved in `top_1000_models.csv` inside the `data` directory.
```bash
python repo_selector.py
```