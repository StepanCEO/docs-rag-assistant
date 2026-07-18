"""OpenAI-backed implementations of the embed and complete callables."""

from openai import OpenAI

from .config import Settings


class OpenAIBackend:
    """Provides ``embed`` and ``complete`` backed by the OpenAI API."""

    def __init__(self, settings: Settings):
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._chat_model = settings.chat_model
        self._embedding_model = settings.embedding_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._embedding_model, input=texts)
        return [item.embedding for item in response.data]

    def complete(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self._chat_model,
            temperature=0,  # factual answers over creativity
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""
