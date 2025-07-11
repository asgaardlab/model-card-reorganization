import re

from transformers import pipeline

from util import helper, path, constants
from util.helper import save_top_model_list
from util.md_processor import remove_codeblock, remove_images, remove_hyperlinks, remove_urls, remove_emojis


def get_language(text: str, pipe: pipeline, top_k: int = 1) -> str:
    predictions = pipe(text, top_k=top_k, truncation=True)
    return predictions[0]['label']


def preprocess_model_card(text: str) -> str:
    text = remove_codeblock(text)
    text = remove_images(text)
    text = remove_urls(text)
    text = remove_emojis(text)
    return text


def get_model_card_language(model_id: str, pipe: pipeline, top_k: int = 1) -> str:
    model_card = helper.get_model_card(model_id)

    preprocessed_model_card = preprocess_model_card(model_card)
    return get_language(preprocessed_model_card, pipe, top_k)


def get_language_detection_model():
    model_ckpt = 'papluca/xlm-roberta-base-language-detection'
    return pipeline('text-classification', model=model_ckpt)


def detect_readmes_language():
    top_models = helper.get_top_model_list(constants.TOP_N)
    language_detection_model = get_language_detection_model()

    print(f'Detecting language for top model cards')
    top_models['model_card_language_roberta'] = top_models['model_id'].apply(
        lambda model_id: get_model_card_language(model_id, language_detection_model))

    print(top_models['model_card_language_roberta'].value_counts())
    save_top_model_list(top_models)


if __name__ == '__main__':
    detect_readmes_language()
