from dataclasses import dataclass
import logging
from typing import Any, Dict, Generator, List, Optional

from numpy.typing import ArrayLike
from transformers.pipelines.feature_extraction import FeatureExtractionPipeline
from transformers.tokenization_utils_base import TokenSpan

from .sentence import Sentence


class Context(Sentence):
    def __init__(
        self,
        text: str,
        model: FeatureExtractionPipeline,
        metadata: Dict[str, Any],
        token: str,
        token_index: int,
    ) -> None:

        super().__init__(text, model, metadata)

        self._token = token
        self._token_index = token_index

    def __repr__(self) -> str:
        return str({"token": self.token, "text": self.text})

    @property
    def token(self) -> str:
        return self._token

    def token_embedding(self) -> Optional[ArrayLike]:
        if token_span := self._find_token():
            return self._aggregate_embeddings(token_span)
        else:
            return None

    def has_word(self) -> bool:
        return self._word_index() is not None

    def _word_index(self) -> Optional[int]:
        return self._encoding.char_to_word(self._token_index)

    def _find_token(self) -> Optional[TokenSpan]:
        if word_index := self._word_index():
            return self._encoding.word_to_tokens(word_index)
        else:
            logging.warning(
                f"Could not find token '{self.token}' at position {self._token_index}"
            )
            return None

    @classmethod
    def contexts(
        cls,
        text: str,
        token: str,
        context_characters: int,
        model: FeatureExtractionPipeline,
        metadata: Dict[str, Any],
    ) -> Generator["Sentence", None, None]:

        i = text.find(token)
        while i >= 0:
            window_radius: int = int(context_characters / 2)
            start = max(0, i - window_radius)
            end = min(len(text), i + len(token) + window_radius)

            yield cls(text[int(start) : int(end)], model, metadata, token, i - start)

            i = text.find(token, i + len(token))
