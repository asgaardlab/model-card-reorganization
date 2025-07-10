import re
from pathlib import Path

from util import path


def remove_yaml_metadata(raw_readme_path: Path, output_readme_path: Path) -> None:
    with open(raw_readme_path, 'r', encoding='utf-8') as original_file:
        original_text = original_file.read()

    if original_text.startswith('---'):
        cleaned_content = re.sub(r'---[\s\S]*?---', '', original_text, count=1)
    else:
        cleaned_content = original_text

    if cleaned_content.strip() != '':
        output_readme_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_readme_path, 'w', encoding='utf-8') as output_file:
            output_file.write(cleaned_content)


def process_readme_files():
    print('Processing readme files (removing YAML metadata)')
    for raw_readme_file in path.RAW_MODEL_CARD_DIRECTORY.iterdir():
        destination_file = path.PREPROCESSED_MODEL_CARD_DIRECTORY / raw_readme_file.name
        remove_yaml_metadata(raw_readme_file, destination_file)

    print(f'{sum(1 for file in path.PREPROCESSED_MODEL_CARD_DIRECTORY.iterdir() if file.is_file())} repositories have model cards after removing YAML metadata')


if __name__ == '__main__':
    process_readme_files()
