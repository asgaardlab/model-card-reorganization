import google.generativeai as genai

from api_clients.vertexai_client import VertexAIClient


class GeminiJsonClient(VertexAIClient):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, api_key)

    def setup_model(self, instruction: str | None, response_schema: dict | None = None):
        genai.configure(api_key=self.api_key)

        generation_config = dict({
            'temperature': 0,
            'top_p': 0.95,
            'top_k': 64,
            'max_output_tokens': 65536,
            'response_mime_type': 'application/json',
            'response_schema': response_schema
        })

        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config,
            system_instruction=instruction
        )
