from abc import ABC, abstractmethod


class LLMBaseClient(ABC):
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        self.model = None

    @abstractmethod
    def setup_model(self, instruction: str | None, response_schema: dict | None):
        pass

    @abstractmethod
    def make_request(self, instruction: str | None, prompt: str, response_schema: dict | None) -> str:
        pass
