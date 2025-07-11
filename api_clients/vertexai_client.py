import google.generativeai as genai

from api_clients.llm_base_client import LLMBaseClient
from util import constants


class VertexAIClient(LLMBaseClient):
    def __init__(self, model_name: str = constants.GEMINI_2_FLASH_THINKING, api_key: str = constants.GEMINI_API_KEY):
        super().__init__(model_name, api_key)

    def setup_model(self, instruction: str | None, response_schema: dict | None = None):
        genai.configure(api_key=self.api_key)

        generation_config = dict({
            'temperature': 0,
            'top_p': 0.95,
            'top_k': 64,
            'max_output_tokens': 65536,
            'response_mime_type': 'text/plain'
        })

        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config,
            system_instruction=instruction
        )

    def make_request(self, instruction, prompt, response_schema) -> str:
        chat_session = self.model.start_chat(history=[])
        response = chat_session.send_message(prompt)

        return response.text
