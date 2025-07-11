from pathlib import Path

from api_clients.gemini_json_client import GeminiJsonClient
from reorganization_verifier.relevance_verifier.base_relevance_verifier import \
    BaseRelevanceVerifier
from util import constants, helper


def get_json_schema() -> dict:
    return {
        'type': 'object',
        'properties': {
            'sections': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'section_name': {
                            'type': 'string',
                            'enum': [
                                'Model Details: Person or organization developing model',
                                'Model Details: Model date',
                                'Model Details: Model version',
                                'Model Details: Model type',
                                'Model Details: Training details',
                                'Model Details: Paper or other resource for more information',
                                'Model Details: Citation details',
                                'Model Details: License',
                                'Model Details: Contact',
                                'Intended Use: Primary intended uses',
                                'Intended Use: Primary intended users',
                                'Intended Use: Out-of-scope uses',
                                'How to Use',
                                'Factors: Relevant factors',
                                'Factors: Evaluation factors',
                                'Metrics: Model performance measures',
                                'Metrics: Decision thresholds',
                                'Metrics: Variation approaches',
                                'Evaluation Data: Datasets',
                                'Evaluation Data: Motivation',
                                'Evaluation Data: Preprocessing',
                                'Training Data: Datasets',
                                'Training Data: Motivation',
                                'Training Data: Preprocessing',
                                'Quantitative Analyses: Unitary results',
                                'Quantitative Analyses: Intersectional results',
                                'Memory or Hardware Requirements: Loading Requirements',
                                'Memory or Hardware Requirements: Deploying Requirements',
                                'Memory or Hardware Requirements: Training or Fine-tuning Requirements',
                                'Ethical Considerations',
                                'Caveats and Recommendations: Caveats',
                                'Caveats and Recommendations: Recommendations'
                            ]
                        },
                        'is_relevant': {
                            'type': 'boolean'
                        },
                        'irrelevant_information': {
                            'type': 'string'
                        }
                    },
                    'required': [
                        'section_name',
                        'is_relevant'
                    ]
                }
            }
        }
    }


class GeminiRelevanceVerifier(BaseRelevanceVerifier):
    def get_save_path(self, model_id: str, next_iteration_no: int = None, sub_dir_name: str = None) -> Path:
        return super().get_save_path(model_id, next_iteration_no, 'gemini_2_5_pro')


if __name__ == '__main__':
    gemini_client = GeminiJsonClient(constants.GEMINI_2_5_PRO, constants.GEMINI_API_KEY)
    relevance_verifier = GeminiRelevanceVerifier(Path('common_prompt_template.md'), get_json_schema(), gemini_client)
    # relevance_verifier.process_single_request('Qwen/Qwen2-VL-7B-Instruct', 5)

    selected_models = helper.get_selected_repos()
    model_ids = selected_models['model_id'].tolist()
    relevance_verifier.process_batch_request(model_ids)
