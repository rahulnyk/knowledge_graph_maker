from ollama import Client
from ..types import LLMClient


class OllamaClient(LLMClient):
    _url: str
    _model: str
    _temperature: float
    _top_p: float

    def __init__(self, model: str = "llama3:8b", temperature=0.2, top_p=1,  url: str = "http://localhost:11434"):
        self._model = model
        self._temperature = temperature
        self._top_p = top_p
        self._url = url
        self._client = Client(host=url)

    def generate(self, user_message: str, system_message: str) -> str:
        options = {
            "top_p": self._top_p,
            "temperature": self._temperature,
        }
        print(user_message)
        response = self._client.generate(model=self._model, system=system_message, prompt=user_message, options=options,
                                   keep_alive=str(5) + "m")
        return response['response']
