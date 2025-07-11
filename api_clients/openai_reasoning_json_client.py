from api_clients.openai_chat_client import OpenAIChatClient
from util import constants


class OpenAIReasoningJsonClient(OpenAIChatClient):
    def __init__(self, model_name: str = constants.O4_MINI, api_key: str = constants.OPENAI_API_KEY,
                 base_url: str = None):
        super().__init__(model_name, api_key)
        self.base_url = base_url

    def make_request(self, instruction, prompt, response_schema: dict | None) -> str:
        messages = []
        if instruction not in [None, '']:
            messages.append({
                'role': 'developer',
                'content': [
                    {
                        'type': 'input_text',
                        'text': instruction
                    }
                ]
            })
        if prompt not in [None, '']:
            messages.append({
                'role': 'user',
                'content': [
                    {
                        'type': 'input_text',
                        'text': prompt
                    }
                ]
            })

        try:
            response = self.model.responses.create(
                model=self.model_name,
                input=messages,
                text={
                    'format': {
                        'type': 'json_schema',
                        'name': 'json_response',
                        'strict': True,
                        'schema': response_schema
                    }
                },
                reasoning={
                    'effort': 'high',
                },
                tools=[],
                store=True
            )
            return response.output_text
        except Exception as e:
            raise RuntimeError(f'Error: {e}')