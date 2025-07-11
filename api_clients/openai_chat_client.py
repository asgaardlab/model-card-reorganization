from openai import OpenAI

from api_clients.llm_base_client import LLMBaseClient
from util import constants


class OpenAIChatClient(LLMBaseClient):
    def __init__(self, model_name: str = constants.GPT_4O_MINI, api_key: str = constants.OPENAI_API_KEY,
                 base_url: str = None):
        super().__init__(model_name, api_key)
        self.base_url = base_url

    def setup_model(self, instruction: str | None, response_schema: dict | None) -> None:
        if self.base_url is not None:
            self.model = OpenAI(base_url=self.base_url, api_key=self.api_key)
        else:
            self.model = OpenAI(api_key=self.api_key)

    def make_request(self, instruction, prompt, response_schema: dict | None) -> str:
        messages = []
        if instruction not in [None, '']:
            messages.append({
                'role': 'system',
                'content': instruction
            })
        messages.append({
            'role': 'user',
            'content': prompt
        })

        try:
            response = self.model.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f'Error: {e}')
